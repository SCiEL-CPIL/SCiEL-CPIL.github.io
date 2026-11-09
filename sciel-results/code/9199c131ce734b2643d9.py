import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try to find a low-rank decomposition
    # First, try rank 55
    for rank in [55, 56, 57, 58, 59, 60, 61, 62, 63]:
        result = try_find_decomposition(n, m, p, rank)
        if result is not None:
            return result
    # Fallback to schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
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

def try_find_decomposition(n, m, p, rank):
    # This is a placeholder for a search algorithm
    # For now, just return None
    return None
# EVOLVE-BLOCK-END
