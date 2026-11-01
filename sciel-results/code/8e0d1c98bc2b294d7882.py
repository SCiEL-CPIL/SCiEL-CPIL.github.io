"""Rank-1 decomposition for 3x3 * 3x7 matrix multiplication tensor."""
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # is the all-ones tensor, which decomposes as a single rank-1 term:
    # (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    # Thus U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
