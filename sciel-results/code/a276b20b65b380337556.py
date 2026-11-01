"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct using explicit basis vectors
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized index generation
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    flat_idx = np.arange(rank)
    
    u_idx = i_idx.flatten() * m + j_idx.flatten()
    v_idx = j_idx.flatten() * p + k_idx.flatten()
    w_idx = k_idx.flatten() * n + i_idx.flatten()
    
    U[u_idx, flat_idx] = 1.0
    V[v_idx, flat_idx] = 1.0
    W[w_idx, flat_idx] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
