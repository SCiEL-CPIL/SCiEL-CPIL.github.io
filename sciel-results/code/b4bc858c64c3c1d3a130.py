import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # Schoolbook decomposition (Rank 54)
    # This is the baseline.
    # We check time before doing heavy work, but schoolbook is fast.
    
    if time.time() > deadline:
        # Return a minimal valid structure if we are out of time?
        # Schoolbook is O(1) construction, so this shouldn't happen.
        pass

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
# EVOLVE-BLOCK-END
