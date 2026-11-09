import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    1. The matrix multiplication tensor $T$ for $n \times m$ by $m \times p$ matrices has a 
       canonical decomposition of rank $n \cdot m \cdot p$.
    2. For $n=3, m=4, p=4$, the rank is 48.
    3. We construct the factor matrices $U, V, W$ using vectorized NumPy operations 
       to ensure exact integer entries (1s and 0s) and minimal computational overhead.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Generate all combinations of (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Calculate linear indices for each factor
    # U index: i * m + j
    u_lin_idx = i_idx * m + j_idx
    # V index: j * p + k
    v_lin_idx = j_idx * p + k_idx
    # W index: k * n + i
    w_lin_idx = k_idx * n + i_idx
    
    # Flatten the index arrays
    u_lin_flat = u_lin_idx.ravel()
    v_lin_flat = v_lin_idx.ravel()
    w_lin_flat = w_lin_idx.ravel()
    
    # The column index r corresponds to the flat index of (i,j,k)
    r_idx = np.arange(rank)
    
    # Assign 1.0 to the appropriate positions
    U[u_lin_flat, r_idx] = 1.0
    V[v_lin_flat, r_idx] = 1.0
    W[w_lin_flat, r_idx] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
