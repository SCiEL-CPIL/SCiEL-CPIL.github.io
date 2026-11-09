"""Matrix multiplication 3,3,7 decomposition using NumPy."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W' containing numpy arrays.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Populate the matrices for the schoolbook algorithm
    # The tensor T is defined such that C_ik = sum_j A_ij B_jk
    # Indexing conventions:
    # A_flat[i*m + j] = A_ij
    # B_flat[j*p + k] = B_jk
    # C_flat[k*n + i] = C_ik  (Column-major for C, as per W[k*n+i])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U selects A_ij
                idx_u = i * m + j
                # V selects B_jk
                idx_v = j * p + k
                # W outputs to C_ik (stored at k*n + i)
                idx_w = k * n + i
                
                U[idx_u, col] = 1.0
                V[idx_v, col] = 1.0
                W[idx_w, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
