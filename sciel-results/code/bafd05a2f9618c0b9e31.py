"""
Matrix multiplication tensor decomposition for 4x5 * 5x6.

Strategy:
1. Use the schoolbook decomposition (rank 120).
2. This is guaranteed to be correct and exact.
3. Attempting to find a lower rank decomposition via random search is unlikely to succeed within the time limit and is not robust.
"""

import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook decomposition
    def schoolbook():
        rank = schoolbook_rank
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
    
    return schoolbook()
