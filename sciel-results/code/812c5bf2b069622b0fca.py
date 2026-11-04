import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25
    
    # Schoolbook construction (rank 125)
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(N)]
    V = [[0] * rank_school for _ in range(M)]
    W = [[0] * rank_school for _ in range(P)]
    
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
