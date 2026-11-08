import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank decomposition for matrix multiplication M_{3,4,5}.
    Uses a vectorized approach to construct the trivial decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Construct U, V, W
    # U: (n*m, rank)
    # Row index for U is i*m + j
    # Col index is the decomposition term index (0 to rank-1)
    # U[row, col] = 1 if term col corresponds to (i,j,k)
    
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
