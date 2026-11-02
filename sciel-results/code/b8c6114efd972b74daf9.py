import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    
    u_rows = i_vals * m + j_vals
    v_rows = j_vals * p + k_vals
    w_rows = k_vals * n + i_vals
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    cols = np.arange(rank)
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists to match the seed program's return type exactly
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
