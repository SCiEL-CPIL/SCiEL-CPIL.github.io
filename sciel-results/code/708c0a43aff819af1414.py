import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We can construct the identity-like decomposition
    # Each rank-1 term corresponds to a unique (i, j, k)
    # U has 1 at (i*m + j)
    # V has 1 at (j*p + k)
    # W has 1 at (k*n + i)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
