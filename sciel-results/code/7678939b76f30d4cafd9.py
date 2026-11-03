import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for vectorized construction
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Map to matrix indices
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Assign 1s to the appropriate positions
    # U[u_idx, col] = 1
    # V[v_idx, col] = 1
    # W[w_idx, col] = 1
    # Since col is just 0..rank-1, we can use advanced indexing
    
    cols = np.arange(rank)
    U[u_idx, cols] = 1
    V[v_idx, cols] = 1
    W[w_idx, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
