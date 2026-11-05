import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 27 Schoolbook Decomposition
    # This is the standard, valid decomposition.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps A_{i,j}
                U[i * m + j, col] = 1
                # V maps B_{j,k}
                V[j * p + k, col] = 1
                # W maps C_{k,i}
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
