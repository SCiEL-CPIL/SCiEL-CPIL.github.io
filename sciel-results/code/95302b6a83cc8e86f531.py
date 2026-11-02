"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.
    
    For M_{3,3,7}, the trivial schoolbook decomposition has rank 3*3*7 = 63.
    We construct this decomposition.
    
    Returns:
        dict: A dictionary containing 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    r_indices = np.arange(rank)
    
    U[u_rows, r_indices] = 1
    V[v_rows, r_indices] = 1
    W[w_rows, r_indices] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
