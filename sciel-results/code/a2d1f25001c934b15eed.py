import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index grids
    idx_i, idx_j, idx_k = np.indices((n, m, p))
    
    # Flatten indices
    i_flat = idx_i.flatten()
    j_flat = idx_j.flatten()
    k_flat = idx_k.flatten()
    
    # Initialize arrays
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Calculate linear indices for rows
    row_u = i_flat * m + j_flat
    row_v = j_flat * p + k_flat
    row_w = k_flat * n + i_flat
    
    # Set values
    # The column index is the rank index (0 to rank-1)
    cols = np.arange(rank)
    
    U[row_u, cols] = 1
    V[row_v, cols] = 1
    W[row_w, cols] = 1
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
