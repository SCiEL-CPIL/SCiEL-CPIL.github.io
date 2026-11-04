import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct U, V, W using vectorized operations for structural difference
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Create indices for the trivial decomposition
    # Each term k in the rank corresponds to a specific (i, j, k_p) triple
    # i in [0, n), j in [0, m), k_p in [0, p)
    
    # Generate all combinations of i, j, k_p
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # U: row index is i*m + j, column index is the term index
    U = np.zeros((n * m, rank), dtype=int)
    rows_u = i_flat * m + j_flat
    cols_u = np.arange(rank)
    U[rows_u, cols_u] = 1
    
    # V: row index is j*p + k_p, column index is the term index
    V = np.zeros((m * p, rank), dtype=int)
    rows_v = j_flat * p + k_flat
    cols_v = np.arange(rank)
    V[rows_v, cols_v] = 1
    
    # W: row index is k_p*n + i, column index is the term index
    # Note: The task specifies W[k*n+i, r]
    W = np.zeros((p * n, rank), dtype=int)
    rows_w = k_flat * n + i_flat
    cols_w = np.arange(rank)
    W[rows_w, cols_w] = 1
    
    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
