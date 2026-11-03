"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses the standard schoolbook algorithm with rank 125.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the 5x5x5 matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W' mapping to lists of lists (matrices).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Pre-allocate matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Fill in the non-zero entries for the schoolbook decomposition
    # The tensor is T = sum_{c=0}^{rank-1} U[:, c] \otimes V[:, c] \otimes W[:, c]
    # For each term c, we pick indices i, j, k such that:
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # This ensures the product sum recovers the matrix multiplication entry (i, k) = sum_j A[i,j]B[j,k]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][c] = 1.0
                V[idx_v][c] = 1.0
                W[idx_w][c] = 1.0
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
