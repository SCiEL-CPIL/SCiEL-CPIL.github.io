import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = m
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            U[idx_u, j] = 1.0
            
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            V[idx_v, j] = 1.0
            
    # W is all ones
    
    return {"U": U, "V": V, "W": W}
