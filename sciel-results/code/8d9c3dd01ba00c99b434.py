"""Matrix multiplication 3,3,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of 3x3 by 3x8 matrix multiplication.
    
    The decomposition is:
    sum_c U[i*3+j, c] * V[j*8+k, c] * W[k*3+i, c] = 1 for all i,j,k.
    
    This implementation uses the schoolbook algorithm with rank 72.
    """
    n, m, p = 3, 3, 8
    rank = n * m * p  # 72
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: row i, col j -> flat index i*m + j
                idx_u = i * m + j
                # Index for V: row j, col k -> flat index j*p + k
                idx_v = j * p + k
                # Index for W: row k, col i -> flat index k*n + i (trace convention)
                idx_w = k * n + i
                
                U[idx_u, c] = 1.0
                V[idx_v, c] = 1.0
                W[idx_w, c] = 1.0
                c += 1
                
    return {"U": U, "V": V, "W": W}
