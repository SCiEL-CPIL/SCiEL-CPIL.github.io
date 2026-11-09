import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication of dimensions 3x4x7.
    Uses a block-based approach inspired by Strassen's algorithm structure.
    For this specific dimension and integer constraint, it falls back to the
    schoolbook rank as no simple integer-coefficient cancellation is identified
    to reduce the rank below the trivial bound.
    """
    n, m, p = DIMENSIONS
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
