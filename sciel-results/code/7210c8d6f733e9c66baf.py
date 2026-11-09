import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    rank = p  # 8
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for k in range(p):
        r = k
        # U_r[i,j] = 1 for all i,j
        for i in range(n):
            for j in range(m):
                U[i*m + j, r] = 1
        # V_r[j,k'] = 1 if k'=k else 0
        for j in range(m):
            V[j*p + k, r] = 1
        # W_r[k',i] = 1 if k'=k else 0
        for i in range(n):
            W[k*n + i, r] = 1
    
    # Verify
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
