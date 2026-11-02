"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 5.0:
                    # Return whatever we have so far, though it might be incomplete
                    # In this simple loop, it's better to just finish if it's fast
                    # 60 iterations is trivial, so this guard is mostly for safety
                    pass
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
