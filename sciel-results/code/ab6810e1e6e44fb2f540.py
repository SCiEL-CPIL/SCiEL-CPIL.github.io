import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: split 4x4 into 2x2 blocks, 4x9 into 2x(4,5)
    # This gives rank <= 4 * rank(2x2 x 2x4) + 4 * rank(2x2 x 2x5)
    # For 2x2 x 2x4: use Strassen (rank 7) for each 2x2x2 subblock, total 14
    # For 2x2 x 2x5: use Strassen for two 2x2x2 and one 2x2x1, total 7+7+1=15
    # Total: 4*14 + 4*15 = 116
    
    # Actually, let me just use the trivial rank-144 for now and try to improve
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
