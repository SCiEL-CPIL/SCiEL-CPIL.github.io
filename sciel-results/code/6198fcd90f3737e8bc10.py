import time
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Returns the schoolbook decomposition of rank n*m*p = 24.
    This is the standard explicit construction and is guaranteed to be valid.
    """
    n, m, p = 2, 3, 4
    rank = n * m * p
    
    # Initialize U, V, W
    # U has shape (n*m, rank) -> (6, 24)
    # V has shape (m*p, rank) -> (12, 24)
    # W has shape (p*n, rank) -> (8, 24)
    
    # Using lists of lists to match the expected interface style, 
    # though numpy could be used internally if converted. 
    # The interface expects lists of lists or arrays with integer/half-integer entries.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # A is n x m. Row i, Col j.
                # B is m x p. Row j, Col k.
                # C is n x p. Row i, Col k.
                
                # U input for A[i,j]
                u_idx = i * m + j
                # V input for B[j,k]
                v_idx = j * p + k
                # W output for C[i,k]
                w_idx = k * n + i
                
                U[u_idx][column] = 1.0
                V[v_idx][column] = 1.0
                W[w_idx][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
