"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n * m * p
    r = n * m * p
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # Vectorized construction for speed and clarity
    # c indexes the rank dimension (0 to r-1)
    c = np.arange(r)
    
    # Extract i, j, k from c
    # c = i * (m * p) + j * p + k
    k = c % p
    j = (c // p) % m
    i = (c // (m * p)) % n
    
    # Compute row indices for U, V, W
    u_idx = i * m + j
    v_idx = j * p + k
    w_idx = k * n + i
    
    # Populate matrices
    U[u_idx, c] = 1.0
    V[v_idx, c] = 1.0
    W[w_idx, c] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
