"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    
    The matrix multiplication tensor T[i,j,k] corresponds to:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We decompose T as sum_c U[:,c] * V[:,c] * W[:,c] where:
    - U has shape (n*m, rank) with U[i*m+j, c]
    - V has shape (m*p, rank) with V[j*p+k, c]  
    - W has shape (p*n, rank) with W[k*n+i, c]
    
    And the tensor identity is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Schoolbook decomposition: rank = n*m*p = 45
    # For each (i,j,k), we have one term:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
