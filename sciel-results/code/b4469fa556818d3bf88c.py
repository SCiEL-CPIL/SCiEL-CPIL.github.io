"""Matrix multiplication 2x4 by 4x8 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Schoolbook decomposition: rank = n*m*p = 64
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 16 x 64
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
