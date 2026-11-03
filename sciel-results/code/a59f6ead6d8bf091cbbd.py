"""Rewritten program for matrix multiplication 2x3 by 3x4.
Uses a vectorized construction approach to define the factor matrices U, V, W.
The rank is the product of dimensions n*m*p = 24.
"""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of the trivial decomposition
    # Create indices for all combinations of i, j, k
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Calculate column indices
    cols = i_idx * m * p + j_idx * p + k_idx
    
    # Calculate row indices for U, V, W
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Assign values using advanced indexing
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
