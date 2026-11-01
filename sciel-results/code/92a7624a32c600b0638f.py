import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all terms
    # i in [0, n), j in [0, m), k in [0, p)
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), 
        np.arange(m), 
        np.arange(p), 
        indexing='ij'
    )
    
    # Flatten the indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Column indices are 0 to rank-1
    cols = np.arange(rank)
    
    # Initialize sparse-like structure or dense matrices
    # Since rank is small (56), dense is fine
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
