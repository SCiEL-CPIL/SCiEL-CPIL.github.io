"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def _construct_rank24():
    """Trivial rank-24 decomposition for 2x3x4 matrix multiplication."""
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 12 x 24
    W = [[0] * rank for _ in range(p * n)]  # 8 x 24
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    return _construct_rank24()
# EVOLVE-BLOCK-END
