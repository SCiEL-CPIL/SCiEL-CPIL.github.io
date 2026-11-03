import numpy as np

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    rank = 180
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for _ in range(50):
        # Solve U
        A = np.einsum('jr,kr->jkr', V, W).reshape(m*p*p*n, rank)
        U = np.linalg.lstsq(A.T, T.reshape(n*m, -1).T, rcond=None)[0].T
        
        # Solve V
        B = np.einsum('ir,kr->ikr', U, W).reshape(n*m*p*n, rank)
        V = np.linalg.lstsq(B.T, T.transpose(1,0,2).reshape(m*p, -1).T, rcond=None)[0].T
        
        # Solve W
        C = np.einsum('ir,jr->ijr', U, V).reshape(n*m*m*p, rank)
        W = np.linalg.lstsq(C.T, T.transpose(2,1,0).reshape(p*n, -1).T, rcond=None)[0].T
    
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
