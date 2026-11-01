import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Naive matrix multiplication decomposition
    # Rank is m
    rank = m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(m):
        # U[i, r] = 1 if j == r
        # U index is i * m + j
        for i in range(n):
            U[i * m + r, r] = 1
            
        # V[j', k, r] = 1 if j' == r
        # V index is j' * p + k
        for k in range(p):
            V[r * p + k, r] = 1
            
        # W[k, i, r] = 1
        # W index is k * n + i
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
