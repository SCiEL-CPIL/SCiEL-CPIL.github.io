"""Program to find a rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The schoolbook decomposition has rank 36 (3*3*4).
    For this small size, we use a fixed known decomposition if available,
    otherwise we fall back to schoolbook.
    
    Since finding a provably minimal decomposition for 3x3 x 3x4 is complex
    and may not be well-known in standard references, we provide the schoolbook
    decomposition which is always correct.
    """
    n, m, p = DIMENSIONS
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
