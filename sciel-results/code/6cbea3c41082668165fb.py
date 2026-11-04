import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        # U: Column r corresponds to intermediate index j=r
        # U[i, j] is 1 if j == r
        for i in range(n):
            U[i * m + r, r] = 1
            
        # V: Column r corresponds to intermediate index j=r
        # V[j, k] is 1 if j == r
        for k in range(p):
            V[r * p + k, r] = 1
            
        # W: Column r contributes to all output entries C[i, k]
        # W[k, i] is 1 for all i, k
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
