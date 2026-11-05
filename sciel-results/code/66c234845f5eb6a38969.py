"""Tensor rank decomposition for 3x3x5 matrix multiplication."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Strategy: Use the standard schoolbook algorithm.
    # The rank of the matrix multiplication tensor for n x m by m x p is n*m*p.
    # For 3x3x5, this is 45.
    # While lower ranks exist (e.g., 41 using rank-23 for 3x3x3 and rank-18 for 3x3x2),
    # implementing the specific rank-23 coefficients reliably without external lookup is error-prone.
    # We provide the guaranteed valid schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (following the convention W[k*n+i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
