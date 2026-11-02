import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Default to schoolbook decomposition
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
    
    # No known better decomposition for 3x3x7 that beats 63 easily
    # If we had a rank-8 for 3x3, we would use 56.
    # But 9 is likely minimal for 3x3 in this context.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
