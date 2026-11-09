"""Matrix multiplication tensor decomposition for M(2,3,10)."""
DIMENSIONS = [2, 3, 10]

import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = n * m * p  # 60
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k in range(p):
        for j in range(m):
            for i in range(n):
                col = k * (n * m) + j * n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
