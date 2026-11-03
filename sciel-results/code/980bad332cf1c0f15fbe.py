"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    
    # We construct a rank-21 decomposition by splitting the 3x7 matrix B into three blocks:
    # B1 (3x3), B2 (3x3), B3 (3x1).
    # We use schoolbook decomposition for each block.
    # Rank(B1) = 9, Rank(B2) = 9, Rank(B3) = 3.
    # Total rank = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_col = 0
    
    # Block 1: k in [0, 1, 2]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][current_col] = 1
                V[j * p + k][current_col] = 1
                W[k * n + i][current_col] = 1
                current_col += 1
                
    # Block 2: k in [3, 4, 5]
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][current_col] = 1
                V[j * p + k][current_col] = 1
                W[k * n + i][current_col] = 1
                current_col += 1
                
    # Block 3: k in [6]
    for i in range(n):
        for j in range(m):
            for k in range(6, 7):
                U[i * m + j][current_col] = 1
                V[j * p + k][current_col] = 1
                W[k * n + i][current_col] = 1
                current_col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
