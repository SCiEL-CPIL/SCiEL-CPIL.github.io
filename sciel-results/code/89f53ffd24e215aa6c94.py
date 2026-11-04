"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Construct the rank-minimal (known) decomposition for 2x4 times 4x6 using NumPy.
    
    Using schoolbook: rank = 2*4*6 = 48.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    # Generate indices for (i, j, k)
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    col_idx = np.arange(rank).reshape((n, m, p))
    
    # Flatten indices for assignment
    cols = col_idx.flatten()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i*m + j, col] = 1
    u_rows = (i_idx * m + j_idx).flatten()
    U[u_rows, cols] = 1.0
    
    # V[j*p + k, col] = 1
    v_rows = (j_idx * p + k_idx).flatten()
    V[v_rows, cols] = 1.0
    
    # W[k*n + i, col] = 1
    w_rows = (k_idx * n + i_idx).flatten()
    W[w_rows, cols] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
