"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of the trivial decomposition
    # Indices:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    # Generate all combinations
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    u_indices = i_vals * m + j_vals
    v_indices = j_vals * p + k_vals
    w_indices = k_vals * n + i_vals
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[u_indices, np.arange(rank)] = 1
    V[v_indices, np.arange(rank)] = 1
    W[w_indices, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
