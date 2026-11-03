"""Optimized tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a low-rank tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    This implementation uses the schoolbook algorithm, which yields a rank
    of n * m * p. For 3x3x7, the rank is 63.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    rank = n * m * p  # 63
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 63
    V = [[0] * rank for _ in range(m * p)]  # 21 x 63
    W = [[0] * rank for _ in range(p * n)]  # 21 x 63
    
    col = 0
    for i in range(n):  # row of A, row of C
        for j in range(m):  # col of A, row of B
            for k in range(p):  # col of B, col of C
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
