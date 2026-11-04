"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The schoolbook algorithm has rank n*m*p = 3*3*7 = 63.
    We return the schoolbook decomposition which is guaranteed exact with integer entries.
    
    The decomposition satisfies:
      For each rank-1 term r, the contribution to C[i,k] is:
        U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
      summed over r, and over j for each (i,k).
    
    In the schoolbook, each term (i,j,k) contributes exactly 1 to C[i,k] via A[i,j]*B[j,k].
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    rank = n * m * p      # 63
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # shape (9, 63)
    V = [[0] * rank for _ in range(m * p)]  # shape (21, 63)
    W = [[0] * rank for _ in range(p * n)]  # shape (21, 63)
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This term contributes to C[i,k] via A[i,j] * B[j,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
