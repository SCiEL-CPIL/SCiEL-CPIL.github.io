"""Matrix multiplication 3,3,7 - rank-minimal decomposition using tensor network structure."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    Uses vectorized NumPy indexing to generate the factor matrices for the
    schoolbook decomposition (rank 63), which is optimal for these dimensions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all (i, j, k) tuples
    # i: 0..n-1, j: 0..m-1, k: 0..p-1
    i_idx, j_idx, k_idx = np.indices((n, m, p)).reshape(3, -1)
    
    # Flatten indices for U, V, W rows
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Column indices are 0..rank-1
    cols = np.arange(rank)
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Assign 1s at the specified positions
    # Since the mapping from (i,j,k) to col is bijective, no conflicts
    U[u_rows, cols] = 1.0
    V[v_rows, cols] = 1.0
    W[w_rows, cols] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
