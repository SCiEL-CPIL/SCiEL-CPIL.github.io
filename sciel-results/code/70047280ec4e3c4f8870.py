"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We split the 4x7 matrix B into B1 (4x4) and B2 (4x3)
    # C = A * B = [A * B1, A * B2]
    
    # Rank for 4x4x4 is 16 (schoolbook)
    # Rank for 4x4x3 is 12 (schoolbook)
    # Total rank = 16 + 12 = 28
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 28
    V = [[0] * rank for _ in range(m * p)] # 28 x 28
    W = [[0] * rank for _ in range(p * n)] # 28 x 28
    
    col = 0
    
    # Part 1: A * B1 (4x4 * 4x4 -> 4x4)
    # B1 corresponds to columns 0..3 of B
    # Indices in V: j * p + k, where j is row of B (0..3), k is col of B (0..3)
    # Indices in W: k * n + i, where k is col of B (0..3), i is row of C (0..3)
    for i in range(n):
        for j in range(m):
            for k in range(4): # 4 columns for B1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: A * B2 (4x4 * 4x3 -> 4x3)
    # B2 corresponds to columns 4..6 of B
    # Indices in V: j * p + k, where j is row of B (0..3), k is col of B (4..6)
    # Indices in W: k * n + i, where k is col of B (4..6), i is row of C (0..3)
    for i in range(n):
        for j in range(m):
            for k in range(4, p): # 3 columns for B2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
