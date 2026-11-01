"""
Matrix multiplication tensor decomposition for dimensions 2, 3, 9.
Returns a valid rank-54 decomposition.
"""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a (2,3) matrix
    by a (3,9) matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    
    # Populate the decomposition
    # The rank-1 terms correspond to the tuples (i, j, k)
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    c = 0
    for i in range(n):
        base_u = i * m
        base_w = i
        for j in range(m):
            u_idx = base_u + j
            v_base = j * p
            for k in range(p):
                v_idx = v_base + k
                w_idx = k * n + base_w
                
                U[u_idx][c] = 1.0
                V[v_idx][c] = 1.0
                W[w_idx][c] = 1.0
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
