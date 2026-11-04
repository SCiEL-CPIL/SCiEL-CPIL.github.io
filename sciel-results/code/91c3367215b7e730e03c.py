import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of schoolbook decomposition
    # This uses broadcasting and array indexing instead of nested loops.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Create index arrays for i, j, k
    i_idx = np.arange(n)[:, None, None]
    j_idx = np.arange(m)[None, :, None]
    k_idx = np.arange(p)[None, None, :]
    
    # Flatten to 1D arrays of indices for the rank dimension
    # c = i * m * p + j * p + k
    c_flat = (i_idx * m * p + j_idx * p + k_idx).flatten()
    
    # Row indices for U: i * m + j
    u_rows = (i_idx * m + j_idx).flatten()
    
    # Row indices for V: j * p + k
    v_rows = (j_idx * p + k_idx).flatten()
    
    # Row indices for W: k * n + i
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Set the 1s
    U[u_rows, c_flat] = 1
    V[v_rows, c_flat] = 1
    W[w_rows, c_flat] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
