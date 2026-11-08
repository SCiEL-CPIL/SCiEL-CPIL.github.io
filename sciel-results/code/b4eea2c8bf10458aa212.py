"""Vectorized construction for matrix multiplication tensor."""
DIMENSIONS = [2, 3, 10]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all (i, j, k) tuples
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Compute row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Compute column indices (rank dimension)
    # Matching the convention: col = k*(n*m) + j*n + i
    cols = k_flat * (n * m) + j_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Populate using advanced indexing
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
