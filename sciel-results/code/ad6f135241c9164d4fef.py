import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # Schoolbook decomposition
    # Rank = n * m * p = 4 * 4 * 7 = 112
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] -> C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Wait, W is p*n. Row index is k*n+i? 
                # Task says: "output factor follows the official trace convention W[k*n+i,r]"
                # So W row index is k*n + i.
                # Let's verify shapes.
                # U: (n*m, rank). Row i*m+j.
                # V: (m*p, rank). Row j*p+k.
                # W: (p*n, rank). Row k*n+i.
                
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
