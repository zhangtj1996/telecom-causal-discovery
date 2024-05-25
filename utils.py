import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import torch
import math
import torch.nn as nn
from tqdm import tqdm, trange


from sklearn.preprocessing import SplineTransformer

spline = SplineTransformer(degree=2, n_knots=100)



import osqp
import numpy as np
from scipy import sparse

def get_Phi_Psi_Mono(x,y,seed=1,k=30,f=np.sin,s=4):
    np.random.seed(seed)
    
    cx=x
    cy=y

    #Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0]).reshape(-1,1)
    Phi = np.concatenate([O,x**1,x**2,x**3,x**4],axis=1).T
    Psi = np.concatenate([O,y**1,y**2,y**3],axis=1).T
    Mono = np.concatenate([O-1,O,2*y,3*y**2],axis=1)

    return Phi,Psi,Mono

def solve_osqp(A,b,v):
    # Define problem data
    P = sparse.csc_matrix(A)
    q = -b.flatten()
    A = sparse.csc_matrix(v.T)
    l = np.array([0])
    u = np.array([0])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.9,rho=0.1,verbose=False,polish=True,scaled_termination=True,max_iter=10000)

    # Solve problem
    res = prob.solve()
    return res.x.reshape(-1,1)

def solve_osqp_mono(A,b,v,Mono):
    # Define problem data
    P = sparse.csc_matrix(A)
    q = -b.flatten()
    A = sparse.csc_matrix(np.vstack([v.T,Mono]))
    l = np.full((Mono.shape[0]+1), -100000)
    u = np.full((Mono.shape[0]+1), 0)
    l[0]=0

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1,rho=0.1,verbose=False,polish=True,scaled_termination=True,max_iter=10000)

    # Solve problem
    res = prob.solve()
    return res.x.reshape(-1,1)

def get_Phi_Psi_kern(x,y,seed,k=20,f=np.sin,s=1/6):
    Kxx=kernel_rbf(x)
    Kyy=kernel_rbf(y)
    return Kxx,Kyy

# def get_Phi_Psi(x,y,seed=1,k=30,f=np.sin,s=4):
#     np.random.seed(seed)
    
#     cx=x
#     cy=y

#     #Add a vector of ones so that w.x + b is just a dot product
#     O = np.ones(cx.shape[0])
#     X = np.column_stack([cx, O])
#     Y = np.column_stack([cy, O])

#     # Random linear projections    
#     Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
#     Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
#     X = np.dot(X, Rx)
#     Y = np.dot(Y, Ry)
#     # Apply non-linear function to random projections
#     fX = np.sin(X)
#     fY = np.sin(Y)

#     Phi=(fX.T)
#     Psi=(fY.T)

#     return Phi,Psi

def get_Phi_Psi(x,y,seed=1,k=30,f=np.sin,s=4):
    np.random.seed(seed)
    
    cx=x
    cy=y

    #Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections    
    Rx1 = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry1 = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X1 = np.dot(X, Rx1)
    Y1 = np.dot(Y, Ry1)
    Rx2 = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry2 = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X2 = np.dot(X, Rx2)
    Y2 = np.dot(Y, Ry2)
    
    
    # Apply non-linear function to random projections
    fX = np.concatenate([np.sin(X1),np.cos(X2)],axis =1)
#     print(fX)
    fY = np.concatenate([np.sin(Y1),np.cos(Y2)],axis =1)


    Phi=(fX.T)
    Psi=(fY.T)

    return Phi,Psi

def update(A,b,v):
    """
    solve 
    min 1/2x^TAx -b^Tx 
    s.t. v^T x=0
    """
    c = np.concatenate([A,v],axis=1)
    d = np.concatenate([v.T,np.array([[0]])],axis=1)
    C = np.concatenate([c,d],axis=0)
    
    x=(np.linalg.inv(C)@(np.append(b,0).reshape(-1,1)))[:-1,:]
    
    return x  

def kernel_rbf(x):
    sigma = np.median(distance.cdist(x.reshape(-1,1), x.reshape(-1,1), "euclidean"))
    return np.exp( -((x-x.T)**2) / (sigma**2) ) # can be linear as well  return x@x.T

def kernel_lin(x):
    return x@x.T

def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Code by: Gary Doran  https://github.com/garydoranjr/rdc
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))

def split_data(data,n_bins):
    sortdata= data[data[:,0].argsort()]
    data_bins=np.split(sortdata,[int((i+1)*sortdata.shape[0]/n_bins) for i in range(n_bins-1)])
    return data_bins


def update_a(v0,v1,P,N):
    P_inv = np.linalg.inv(P)
    lam1 = (v1.T@P_inv@v0)/(v1.T@P_inv@v1)
    lam2 = np.sqrt((v0-lam1*v1).T@P_inv@(v0-lam1*v1)/4/N)
    
    
    alpha = P_inv@(v0-lam1*v1)/2/lam2
    if -alpha.T@v0>alpha.T@v0:
        alpha = -alpha
    
    return alpha
  


def MCPNL(A, B, plot = False, test_mode=False,lamb=5):

    n_epoch=150
    
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    
    PsiPsiT = Psi@Psi.T
    PhiPhiT = Phi@Phi.T
    PhiPsiT = Phi@Psi.T 
    midA = (Phi@H@Kxx@H@Phi.T)
    midB = Phi@H@Kxx@H@Psi.T
    midC = (Psi@H@Kxx@H@Psi.T)
    for epoch in range(n_epoch):
        A = (beta.T@PsiPsiT@beta) * (PhiPhiT)/N/N + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
        b = PhiPsiT@beta/N + 2*lamb* midB@beta/N**2
        alpha =  update(A,b,v1)

        A = (alpha.T@PhiPhiT@alpha) * (PsiPsiT)/N/N + lamb* (midC/N**2)+EPSILON *np.eye(Psi.shape[0])
        b = PhiPsiT.T@alpha/N + 2*lamb* midB.T@alpha/N**2
        beta =  update(A,b,v2)

        #loss=(alpha.T@Phi@Phi.T@alpha) * (beta.T@Psi@Psi.T@beta)/2/N/N-alpha.T@Phi@Psi.T@beta/N 
#         print(loss)
        #\
        #lamb*(alpha.T@Kxx@alpha+beta.T@Kyy@beta)
        #HSIC=(alpha.T@Phi@H@Kxx@H@Phi.T@alpha+beta.T@Psi@H@Kxx@H@Psi.T@beta-2* beta.T@Psi@H@Kxx@H@Phi.T@alpha)/N**2
#     print("HSIC_RBF:", np.trace(Kxx@H@kernel_rbf(Phi.T@alpha-Psi.T@beta)@H)/N**2)
#     print("HSIC:",HSIC)
    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
    
    test_rdc = rdc(X,(Psi.T@beta-Phi.T@alpha),n=10)
    if test_mode == True:
        test_rdc = rdc(X_te,(Psi_te.T@beta-Phi_te.T@alpha),n=10)
        print("test_rdc",test_rdc)
    return test_rdc


def minHSIC_lin(A, B, plot = False, test_mode=False,lamb=5):

    n_epoch=100
    
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    
    PsiPsiT = Psi@Psi.T
    PhiPhiT = Phi@Phi.T
    PhiPsiT = Phi@Psi.T 
    midA = (Phi@H@Kxx@H@Phi.T)
    midB = Phi@H@Kxx@H@Psi.T
    midC = (Psi@H@Kxx@H@Psi.T)
    for epoch in range(n_epoch):
        A = + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
        b =  + 2*lamb* midB@beta/N**2
        alpha =  update(A,b,v1)

        A =  + lamb* (midC/N**2)+EPSILON *np.eye(Psi.shape[0])
        b = + 2*lamb* midB.T@alpha/N**2
        beta =  update(A,b,v2)

        #loss=(alpha.T@Phi@Phi.T@alpha) * (beta.T@Psi@Psi.T@beta)/2/N/N-alpha.T@Phi@Psi.T@beta/N 
#         print(loss)
        #\
        #lamb*(alpha.T@Kxx@alpha+beta.T@Kyy@beta)
        #HSIC=(alpha.T@Phi@H@Kxx@H@Phi.T@alpha+beta.T@Psi@H@Kxx@H@Psi.T@beta-2* beta.T@Psi@H@Kxx@H@Phi.T@alpha)/N**2
#     print("HSIC_RBF:", np.trace(Kxx@H@kernel_rbf(Phi.T@alpha-Psi.T@beta)@H)/N**2)
#     print("HSIC:",HSIC)
    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
    
    test_rdc = rdc(X,(Psi.T@beta-Phi.T@alpha),n=10)
    if test_mode == True:
        test_rdc = rdc(X_te,(Psi_te.T@beta-Phi_te.T@alpha),n=10)
        print("test_rdc",test_rdc)
    return test_rdc






def MCPNL_tnLam(A, B, plot = False, test_mode=True,lamb=5):
# tune lamb
    n_epoch=300

    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)

    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 

    Kxx=kernel_rbf(X)


    ind_score_min = 2
    
    for lamb in [0.1,1,5,10,20,50]:
#         print(lamb)


        #init
        Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)

        np.random.seed(0)
        alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
        beta = np.random.randint(0,150,size=(Psi.shape[0],1))

        v1 = np.sum(Phi,1).reshape(-1,1)
        v2 = np.sum(Psi,1).reshape(-1,1)    

        H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])

        PsiPsiT = Psi@Psi.T
        PhiPhiT = Phi@Phi.T
        PhiPsiT = Phi@Psi.T 
        midA = (Phi@H@Kxx@H@Phi.T)
        midB = Phi@H@Kxx@H@Psi.T
        midC = (Psi@H@Kxx@H@Psi.T)
        for epoch in range(n_epoch):
            A = (beta.T@PsiPsiT@beta) * (PhiPhiT)/N/N + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
            b = PhiPsiT@beta/N + 2*lamb* midB@beta/N**2
            alpha =  update(A,b,v1)

            A = (alpha.T@PhiPhiT@alpha) * (PsiPsiT)/N/N + lamb* (midC/N**2)+EPSILON *np.eye(Psi.shape[0])
            b = PhiPsiT.T@alpha/N + 2*lamb* midB.T@alpha/N**2
            beta =  update(A,b,v2)

            #loss=(alpha.T@Phi@Phi.T@alpha) * (beta.T@Psi@Psi.T@beta)/2/N/N-alpha.T@Phi@Psi.T@beta/N 
    #         print(loss)
            #\
            #lamb*(alpha.T@Kxx@alpha+beta.T@Kyy@beta)
            #HSIC=(alpha.T@Phi@H@Kxx@H@Phi.T@alpha+beta.T@Psi@H@Kxx@H@Psi.T@beta-2* beta.T@Psi@H@Kxx@H@Phi.T@alpha)/N**2
    #     print("HSIC_RBF:", np.trace(Kxx@H@kernel_rbf(Phi.T@alpha-Psi.T@beta)@H)/N**2)
    #     print("HSIC:",HSIC)
        if plot ==True:
            plt.figure() 
            plt.scatter(X,Y)
            plt.scatter(X,-Phi.T@alpha)
            plt.figure()
            plt.scatter(X,Phi.T@alpha-Psi.T@beta)
            plt.figure()
            plt.scatter(Phi.T@alpha,Psi.T@beta)
            plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
            plt.figure()
            plt.scatter(Y,Psi.T@beta)
            plt.show()


        if test_mode == True:
            test_rdc = rdc(X_te,(Psi_te.T@beta-Phi_te.T@alpha),n=10)
            if test_rdc < ind_score_min:
                ind_score_min = test_rdc 
        else:
            test_rdc = rdc(X,(Psi.T@beta-Phi.T@alpha),n=10)
            if test_rdc < ind_score_min:
                ind_score_min = test_rdc             
            #print("test_rdc",test_rdc)
    return ind_score_min



def MCPNL_mono(A, B, plot = True, test_mode=False,lamb=5):

    n_epoch=400
    
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi,Mono=get_Phi_Psi_Mono(X,Y,k=30,s=4)  
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    
    PsiPsiT = Psi@Psi.T
    PhiPhiT = Phi@Phi.T
    PhiPsiT = Phi@Psi.T 
    midA = (Phi@H@Kxx@H@Phi.T)
    midB = Phi@H@Kxx@H@Psi.T
    midC = (Psi@H@Kxx@H@Psi.T)
    for epoch in range(n_epoch):
        A = (beta.T@PsiPsiT@beta) * (PhiPhiT)/N/N + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
        b = PhiPsiT@beta/N + 2*lamb* midB@beta/N**2
        alpha =  update(A,b,v1)

        A = (alpha.T@PhiPhiT@alpha) * (PsiPsiT)/N/N + lamb* (midC/N**2)+EPSILON *np.eye(Psi.shape[0])
        b = PhiPsiT.T@alpha/N + 2*lamb* midB.T@alpha/N**2
        randind = np.random.choice(range(N), size=500, replace=False)       
        beta = solve_osqp_mono(A,b,v2,Mono[randind])

        #loss=(alpha.T@Phi@Phi.T@alpha) * (beta.T@Psi@Psi.T@beta)/2/N/N-alpha.T@Phi@Psi.T@beta/N 
#         print(loss)
        #\
        #lamb*(alpha.T@Kxx@alpha+beta.T@Kyy@beta)
        #HSIC=(alpha.T@Phi@H@Kxx@H@Phi.T@alpha+beta.T@Psi@H@Kxx@H@Psi.T@beta-2* beta.T@Psi@H@Kxx@H@Phi.T@alpha)/N**2
#     print("HSIC_RBF:", np.trace(Kxx@H@kernel_rbf(Phi.T@alpha-Psi.T@beta)@H)/N**2)
#     print("HSIC:",HSIC)
    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
    
    test_rdc = rdc(X,(Psi.T@beta-Phi.T@alpha),n=10)
    if test_mode == True:
        test_rdc = rdc(X_te,(Psi_te.T@beta-Phi_te.T@alpha),n=10)
        print("test_rdc",test_rdc)
    return test_rdc



def MCPNL_HSIC_RBF(A, B, plot = True, test_mode=False,lamb=5):

    n_epoch=150
    
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    
    PsiPsiT = Psi@Psi.T
    PhiPhiT = Phi@Phi.T
    PhiPsiT = Phi@Psi.T 
    midA = (Phi@H@Kxx@H@Phi.T)
    midB = Phi@H@Kxx@H@Psi.T
    midC = (Psi@H@Kxx@H@Psi.T)
    for epoch in range(n_epoch):
        A = (beta.T@PsiPsiT@beta) * (PhiPhiT)/N/N + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
        b = PhiPsiT@beta/N + 2*lamb* midB@beta/N**2
        alpha =  update(A,b,v1)

        A = (alpha.T@PhiPhiT@alpha) * (PsiPsiT)/N/N + lamb* (midC/N**2)+EPSILON *np.eye(Phi.shape[0])
        b = PhiPsiT.T@alpha/N + 2*lamb* midB.T@alpha/N**2
        beta =  update(A,b,v2)

        #loss=(alpha.T@Phi@Phi.T@alpha) * (beta.T@Psi@Psi.T@beta)/2/N/N-alpha.T@Phi@Psi.T@beta/N 
#         print(loss)
        #\
        #lamb*(alpha.T@Kxx@alpha+beta.T@Kyy@beta)
        #HSIC=(alpha.T@Phi@H@Kxx@H@Phi.T@alpha+beta.T@Psi@H@Kxx@H@Psi.T@beta-2* beta.T@Psi@H@Kxx@H@Phi.T@alpha)/N**2
#     print("HSIC_RBF:", np.trace(Kxx@H@kernel_rbf(Phi.T@alpha-Psi.T@beta)@H)/N**2)
#     print("HSIC:",HSIC)
    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
    
    test_HSIC_score = HSIC_score(X,(Psi.T@beta-Phi.T@alpha),"RBF")
    if test_mode == True:
        test_HSIC_score = HSIC_score(X_te,(Psi_te.T@beta-Phi_te.T@alpha),"RBF")
        print("test_HSIC_score",test_HSIC_score)
    return test_HSIC_score

def MCPNL_HSIC_RBF_test(A, B, plot = False, test_mode=False,lamb=5):

    n_epoch=150
    
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    
    PsiPsiT = Psi@Psi.T
    PhiPhiT = Phi@Phi.T
    PhiPsiT = Phi@Psi.T 
    midA = (Phi@H@Kxx@H@Phi.T)
    midB = Phi@H@Kxx@H@Psi.T
    midC = (Psi@H@Kxx@H@Psi.T)
    for epoch in range(n_epoch):
        A = (beta.T@PsiPsiT@beta) * (PhiPhiT)/N/N + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
        b = PhiPsiT@beta/N + 2*lamb* midB@beta/N**2
        alpha =  update(A,b,v1)

        A = (alpha.T@PhiPhiT@alpha) * (PsiPsiT)/N/N + lamb* (midC/N**2)+EPSILON *np.eye(Phi.shape[0])
        b = PhiPsiT.T@alpha/N + 2*lamb* midB.T@alpha/N**2
        beta =  update(A,b,v2)


    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
        
    testStat, thresh = hsic_gam(X, (Psi.T@beta-Phi.T@alpha), alph = 0.1)
    #print(testStat,thresh)
    if testStat< thresh:
        test_HSIC_score = 1
    else:
        test_HSIC_score = 0
    
#    test_HSIC_score = HSIC_score(X,(Psi.T@beta-Phi.T@alpha),"RBF")
    if test_mode == True:
        test_HSIC_score = HSIC_score(X_te,(Psi_te.T@beta-Phi_te.T@alpha),"RBF")
        print("test_HSIC_score",test_HSIC_score)
    return test_HSIC_score


# def MCPNL_HSIC(A, B, plot = False, test_mode=False,lamb=5):

#     n_epoch=100
    
#     EPSILON=1e-12
#     N = len(A)

#     x = A.reshape(-1, 1) # single feature
#     y = B.reshape(-1, 1)
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#     scaler_x.fit(x)
#     scaler_y.fit(y)
#     X = scaler_x.transform(x)
#     Y = scaler_y.transform(y)
    
#     if test_mode == True:
#         N= int(0.8*N)
#         X_te,Y_te = X[N:],Y[N:]
#         X,Y = X[:N], Y[:N]
#         Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
#     Kxx=kernel_rbf(X)

#     #init
#     Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
#     np.random.seed(0)
#     alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
#     beta = np.random.randint(0,150,size=(Psi.shape[0],1))

#     v1 = np.sum(Phi,1).reshape(-1,1)
#     v2 = np.sum(Psi,1).reshape(-1,1)    

#     H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    
#     PsiPsiT = Psi@Psi.T
#     PhiPhiT = Phi@Phi.T
#     PhiPsiT = Phi@Psi.T 
#     midA = (Phi@H@Kxx@H@Phi.T)
#     midB = Phi@H@Kxx@H@Psi.T
#     midC = (Psi@H@Kxx@H@Psi.T)
#     for epoch in range(n_epoch):
#         A = (beta.T@PsiPsiT@beta) * (PhiPhiT)/N/N + lamb* (midA/N**2)+EPSILON *np.eye(Phi.shape[0])
#         b = PhiPsiT@beta/N + 2*lamb* midB@beta/N**2
#         alpha =  update(A,b,v1)

#         A = (alpha.T@PhiPhiT@alpha) * (PsiPsiT)/N/N + lamb* (midC/N**2)+EPSILON *np.eye(Phi.shape[0])
#         b = PhiPsiT.T@alpha/N + 2*lamb* midB.T@alpha/N**2
#         beta =  update(A,b,v2)

#         #loss=(alpha.T@Phi@Phi.T@alpha) * (beta.T@Psi@Psi.T@beta)/2/N/N-alpha.T@Phi@Psi.T@beta/N 
# #         print(loss)
#         #\
#         #lamb*(alpha.T@Kxx@alpha+beta.T@Kyy@beta)
#         #HSIC=(alpha.T@Phi@H@Kxx@H@Phi.T@alpha+beta.T@Psi@H@Kxx@H@Psi.T@beta-2* beta.T@Psi@H@Kxx@H@Phi.T@alpha)/N**2
# #     print("HSIC_RBF:", np.trace(Kxx@H@kernel_rbf(Phi.T@alpha-Psi.T@beta)@H)/N**2)
# #     print("HSIC:",HSIC)
#     if plot ==True:
#         plt.figure() 
#         plt.scatter(X,Y)
#         plt.scatter(X,-Phi.T@alpha)
#         plt.figure()
#         plt.scatter(X,Phi.T@alpha-Psi.T@beta)
#         plt.figure()
#         plt.scatter(Phi.T@alpha,Psi.T@beta)
#         plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
#         plt.figure()
#         plt.scatter(Y,Psi.T@beta)
#         plt.show()
        
#     Resi = Psi.T@beta-Phi.T@alpha

#     dist_matrix = abs(X-X.T)
#     np.fill_diagonal(dist_matrix, np.nan)
#     sigma1 = np.nanmedian(dist_matrix)

#     dist_matrix = abs(Resi-Resi.T)
#     np.fill_diagonal(dist_matrix, np.nan)
#     sigma2 = np.nanmedian(dist_matrix)
#     test_HSIC = HSIC_score(X,Resi,sigma1,sigma2,"RQ")
  
#     return test_HSIC


def ACE(A, B, plot = False, test_mode=False):

    n_epoch=150
    lamb=0
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    PhiPhiT = (Phi@Phi.T)
    PhiPsiT= Phi@Psi.T
    PsiPsiT = (Psi@Psi.T)
    for epoch in range(n_epoch):
        A = PhiPhiT/N + EPSILON*np.eye(Phi.shape[0]) 
        b = PhiPsiT@beta/N 
        alpha =  update(A,b,v1)
        A =  PsiPsiT/N +EPSILON*np.eye(Phi.shape[0])
        b = PhiPsiT.T@alpha/N 

        beta =  update(A,b,v2)
        beta = beta / np.sqrt(beta.T@PsiPsiT@beta/N)

    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
    
    test_rdc = rdc(X,(Psi.T@beta-Phi.T@alpha),n=10)
    if test_mode == True:
        test_rdc = rdc(X_te,(Psi_te.T@beta-Phi_te.T@alpha),n=10)

    return test_rdc



def ACE_HSIC(A, B, plot = False, test_mode=False):

    n_epoch=150
    lamb=0
    EPSILON=1e-12
    N = len(A)

    x = A.reshape(-1, 1) # single feature
    y = B.reshape(-1, 1)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x)
    scaler_y.fit(y)
    X = scaler_x.transform(x)
    Y = scaler_y.transform(y)
    
    if test_mode == True:
        N= int(0.8*N)
        X_te,Y_te = X[N:],Y[N:]
        X,Y = X[:N], Y[:N]
        Phi_te,Psi_te=get_Phi_Psi(X_te,Y_te,seed=5,k=30,s=4) 
    
    Kxx=kernel_rbf(X)

    #init
    Phi,Psi=get_Phi_Psi(X,Y,seed=5,k=30,s=4)
    
    np.random.seed(0)
    alpha = np.random.randint(0,150,size=(Phi.shape[0],1))
    beta = np.random.randint(0,150,size=(Psi.shape[0],1))

    v1 = np.sum(Phi,1).reshape(-1,1)
    v2 = np.sum(Psi,1).reshape(-1,1)    

    H = np.eye(N)-1/N*np.ones([N,1])@np.ones([1,N])
    PhiPhiT = (Phi@Phi.T)
    PhiPsiT= Phi@Psi.T
    PsiPsiT = (Psi@Psi.T)
    for epoch in range(n_epoch):
        A = PhiPhiT/N + EPSILON*np.eye(Phi.shape[0]) 
        b = PhiPsiT@beta/N 
        alpha =  update(A,b,v1)
        A =  PsiPsiT/N +EPSILON*np.eye(Phi.shape[0])
        b = PhiPsiT.T@alpha/N 

        beta =  update(A,b,v2)
        beta = beta / np.sqrt(beta.T@PsiPsiT@beta/N)

    if plot ==True:
        plt.figure() 
        plt.scatter(X,Y)
        plt.scatter(X,-Phi.T@alpha)
        plt.figure()
        plt.scatter(X,Phi.T@alpha-Psi.T@beta)
        plt.figure()
        plt.scatter(Phi.T@alpha,Psi.T@beta)
        plt.plot(np.arange(-2,2,0.01),np.arange(-2,2,0.01))
        plt.figure()
        plt.scatter(Y,Psi.T@beta)
        plt.show()
    
    test_HSIC_score = HSIC_score(X,(Psi.T@beta-Phi.T@alpha),"RBF")
    if test_mode == True:
        test_HSIC_score = HSIC_score(X_te,(Psi_te.T@beta-Phi_te.T@alpha),"RBF")
        print("test_HSIC_score",test_HSIC_score)
    return test_HSIC_score


















############################ HSIC#########################################
def RBF_kernel(x,sigma):
    return torch.exp(-(x-x.T)**2/(sigma**2))

def RQ_kernel(x,sigma):
    alpha=0.5
    return (1+(x-x.T)**2/(2*alpha*sigma**2))**(-alpha)

def linear_kernel(x):
    return x@x.T

def HSIC_score(x,y,kernel_name):
    dist_matrix = abs(x-x.T)
    np.fill_diagonal(dist_matrix, np.nan)
    sigma1 = np.nanmedian(dist_matrix)
    dist_matrix = abs(y-y.T)
    np.fill_diagonal(dist_matrix, np.nan)
    sigma2 = np.nanmedian(dist_matrix)    
     
    n = len(x)
    
    x,y =torch.Tensor(x),torch.Tensor(y)
    H= torch.eye(n)-1/n*torch.ones(n,1)@torch.ones(1,n)
    
    if kernel_name == "RQ":        
        K = RQ_kernel(x,sigma1)
        L = RQ_kernel(y,sigma2)
        HSIC = (1/n**2)*torch.trace(K@H@L@H) 
        return HSIC 
        
    if kernel_name == "RBF":
        K = RBF_kernel(x,sigma1)
        L = RBF_kernel(y,sigma2)
        HSIC = (1/n**2)*torch.trace(K@H@L@H) 
        return HSIC 

    if kernel_name == 'linear':
        K = linear_kernel(x)
        L = linear_kernel(y)
        HSIC = (1/n**2)*torch.trace(K@H@L@H) 
        return HSIC
 #####################################################################