"""Program using Vectorized Kronecker Construction for 3x4x4 matrix multiplication."""

import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-48 exact tensor decomposition for multiplying 
    a 3x4 matrix by a 4x4 matrix using a vectorized Kronecker product approach.
    
    The output is a dictionary with keys 'U', 'V', 'W'.
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)
    W: shape (p*n, rank) = (12, 48)
    
    The decomposition follows the convention:
    C_{i,k} = sum_{r} U_{i*m+j} V_{j*p+k} W_{k*n+i}
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for i, j, k
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Calculate row indices for U, V, W
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Set non-zero entries using advanced indexing
    # This is a single vectorized operation instead of a triple loop
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
