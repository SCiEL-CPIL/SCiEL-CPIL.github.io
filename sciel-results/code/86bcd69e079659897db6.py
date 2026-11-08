"""Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (as per trace convention W[k*n+i, r])
                
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
