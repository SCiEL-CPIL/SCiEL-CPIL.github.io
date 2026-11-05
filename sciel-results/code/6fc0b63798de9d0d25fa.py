"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The schoolbook rank is 3 * 3 * 7 = 63.
    We construct the schoolbook decomposition which is guaranteed to be correct.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 3, 7
    rank_schoolbook = n * m * p  # 63

    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
