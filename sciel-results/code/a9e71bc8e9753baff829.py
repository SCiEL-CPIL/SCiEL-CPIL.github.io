import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-60 exact decomposition of the matrix-multiplication
    tensor for 2x5 * 5x6 -> 2x6 using a vectorized index construction.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Generate all combinations of indices (i, j, k)
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten them to create a list of (i, j, k) tuples for each rank term
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    # U row: i*m + j
    u_rows = i_flat * m + j_flat
    # V row: j*p + k
    v_rows = j_flat * p + k_flat
    # W row: k*n + i
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Each rank term r corresponds to the r-th element in the flattened arrays
    # The column index is just 0 to rank-1
    cols = np.arange(rank)
    
    # Set the values
    U[u_rows, cols] = 1.0
    V[v_rows, cols] = 1.0
    W[w_rows, cols] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
