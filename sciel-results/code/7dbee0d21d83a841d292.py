"""Matrix multiplication 3x3 x 3x6 - rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal decomposition for 3x3 * 3x6 matrix multiplication.
    
    Uses schoolbook decomposition with rank n*m*p = 54.
    """
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
