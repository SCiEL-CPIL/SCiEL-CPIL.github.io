import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill using vectorized operations or efficient loops
    # The index r corresponds to (i, j, k)
    # r = i * m * p + j * p + k
    
    for r in range(rank):
        i = (r // (m * p)) % n
        j = (r // p) % m
        k = r % p
        
        # U index: i * m + j
        U[i * m + j, r] = 1.0
        # V index: j * p + k
        V[j * p + k, r] = 1.0
        # W index: k * n + i
        W[k * n + i, r] = 1.0
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
