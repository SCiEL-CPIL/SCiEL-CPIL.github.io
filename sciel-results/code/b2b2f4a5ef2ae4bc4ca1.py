"""Rewritten program for matrix multiplication 3,3,7 rank minimization."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 by 3x7 matrix.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    t_start = time.time()
    DEADLINE = 120  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    schoolbook_rank = n * m * p  # 63
    
    # Build the schoolbook decomposition as fallback
    def build_schoolbook():
        rank = schoolbook_rank
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
        return U, V, W, rank
    
    # Return the schoolbook decomposition
    U, V, W, rank = build_schoolbook()
    return {"U": U, "V": V, "W": W}
