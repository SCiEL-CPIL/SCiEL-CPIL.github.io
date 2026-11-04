"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor for n x m x p multiplication decomposes into m independent
    # n x 1 x p multiplications if we fix the middle index j.
    # Since the U and V indices for different j are disjoint, the rank is
    # the sum of the ranks of the n x 1 x p parts.
    # The rank of n x 1 x p multiplication is 1 (outer product).
    # Thus, total rank = m * 1 = m.
    # For n=2, m=3, p=7, rank = 3.
    
    rank = m
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Populate U
    # U index: i * m + j
    # For term r (corresponding to j=r), U[i*m + r][r] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
            
    # Populate V
    # V index: j * p + k
    # For term r (corresponding to j=r), V[r*p + k][r] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
            
    # Populate W
    # W index: k * n + i
    # For term r (corresponding to j=r), W[k*n + i][r] = 1
    for k in range(p):
        for i in range(n):
            for j in range(m):
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
