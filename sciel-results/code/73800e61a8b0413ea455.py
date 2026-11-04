import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # Each j-slice has rank 1
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for j in range(m):
        c = j
        for i in range(n):
            U[i * m + j, c] = 1
        for k in range(p):
            V[j * p + k, c] = 1
            for i in range(n):
                W[k * n + i, c] = 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
