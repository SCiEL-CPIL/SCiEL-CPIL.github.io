"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Full Schoolbook Decomposition
    # Rank = n * m * p = 4 * 4 * 5 = 80
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 16 x 80
    V = [[0] * rank for _ in range(m * p)]  # 20 x 80
    W = [[0] * rank for _ in range(p * n)]  # 20 x 80
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][col] = 1
                # V index: j * p + k
                V[j * p + k][col] = 1
                # W index: k * n + i (Trace convention)
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
