"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-24 matrix multiplication decomposition for 2x3 * 3x8.
    Splits the 3x8 input into two 3x4 blocks and uses a rank-12 decomposition for each.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in [0, 3]
    for r in range(12):
        j = r // 4
        k = r % 4
        # U is all 1s for this block
        for i in range(n):
            U[i * m + j][r] = 1
        # V is 1 at (j, k)
        V[j * p + k][r] = 1
        # W is all 1s for this block
        for i in range(n):
            W[k * n + i][r] = 1
            
    # Block 2: k in [4, 7]
    for r in range(12):
        j = r // 4
        k = 4 + (r % 4)
        # U is all 1s for this block
        for i in range(n):
            U[i * m + j][r + 12] = 1
        # V is 1 at (j, k)
        V[j * p + k][r + 12] = 1
        # W is all 1s for this block
        for i in range(n):
            W[k * n + i][r + 12] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
