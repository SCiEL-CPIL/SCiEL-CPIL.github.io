import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Rank calculation
    # Split B (3x7) into B1 (3x3) and B2 (3x4)
    # Part 1: A (3x3) * B1 (3x3) -> C1 (3x3). Rank 9.
    # Part 2: A (3x3) * B2 (3x4) -> C2 (3x4). Rank 36.
    # Total Rank = 45.
    
    rank1 = n * m * 3
    rank2 = n * m * (p - 3)
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k in [0, 3)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: k in [3, 7)
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
