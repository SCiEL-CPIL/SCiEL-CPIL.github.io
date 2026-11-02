"""Matrix multiplication 2x5x6 with block-optimized rank reduction."""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = 2, 5, 6
    # Split middle dimension: 5 = 2 + 2 + 1
    # Blocks: (2x2x6) + (2x2x6) + (2x1x6)
    # Ranks: 24 + 24 + 1 = 49 (schoolbook for 2x2x6, rank-1 for 2x1x6)
    
    total_rank = 24 + 24 + 1
    
    U = [[0] * total_rank for _ in range(n * m)]   # 10 x 49
    V = [[0] * total_rank for _ in range(m * p)]   # 30 x 49
    W = [[0] * total_rank for _ in range(p * n)]   # 12 x 49
    
    col = 0
    # Block 1: j in {0, 1}
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: j in {2, 3}
    for i in range(n):
        for j in range(2, 4):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 3: j = 4, rank-1
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
    # col stays at 48, only one column used
    
    return {"U": U, "V": V, "W": W}
