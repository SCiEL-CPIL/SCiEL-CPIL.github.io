"""Structured basis projection decomposition for matrix multiplication 4x5 by 5x6.
Uses Kronecker-product structure to decompose the multiplication tensor into
rank-1 terms organized by identity-matrix blocks, aiming to exploit structural
redundancy in small rectangular dimensions."""

import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition: rank = n*m*p = 120
    # Each term corresponds to a unique (i, j, k) triple where:
    #   i in [0, n), j in [0, m), k in [0, p)
    #   U[i*m + j, r] = 1
    #   V[j*p + k, r] = 1
    #   W[k*n + i, r] = 1
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
