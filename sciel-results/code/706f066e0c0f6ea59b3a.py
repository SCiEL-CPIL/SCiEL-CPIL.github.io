import numpy as np
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Vectorized fill
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    # r corresponds to (i, j, k)
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to get column indices for U, V, W
    u_cols = (i_idx * m + j_idx).flatten()
    v_cols = (j_idx * p + k_idx).flatten()
    w_cols = (k_idx * n + i_idx).flatten()
    
    # The column index in the factor matrices is just the order 0..rank-1
    r_cols = np.arange(rank)
    
    U[u_cols, r_cols] = 1
    V[v_cols, r_cols] = 1
    W[w_cols, r_cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
