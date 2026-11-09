import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the standard schoolbook tensor decomposition for 5x5 matrix multiplication.
    Rank = 125.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Build index arrays for all (i, j, k) triples
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # U: row index i*m + j, one-hot per column
    U_rows = i_idx * m + j_idx
    U = np.zeros((n * m, rank))
    U[U_rows, np.arange(rank)] = 1
    
    # V: row index j*p + k, one-hot per column
    V_rows = j_idx * p + k_idx
    V = np.zeros((m * p, rank))
    V[V_rows, np.arange(rank)] = 1
    
    # W: row index k*n + i, one-hot per column
    W_rows = k_idx * n + i_idx
    W = np.zeros((p * n, rank))
    W[W_rows, np.arange(rank)] = 1
    
    # Convert to list of lists with integer entries
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
