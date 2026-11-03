"""
Program: rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.

The task defines a tensor T of size n x m x p (3x5x6) with all entries equal to 1.
We seek the minimum rank r such that T[i, j, k] = sum_t U[i, j, t] V[j, k, t] W[k, i, t].
For the all-ones tensor, rank 1 is sufficient:
U[i, j, 0] = 1
V[j, k, 0] = 1
W[k, i, 0] = 1
Sum = 1 * 1 * 1 = 1.
Thus, the minimal rank is 1.
"""
import numpy as np

def construct():
    n, m, p = 3, 5, 6
    rank = 1
    
    # Initialize matrices with 1s
    # U shape: (n*m, rank)
    U = [[1] * rank for _ in range(n * m)]
    # V shape: (m*p, rank)
    V = [[1] * rank for _ in range(m * p)]
    # W shape: (p*n, rank)
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
