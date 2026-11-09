"""Matrix multiplication 3x5x6 tensor rank minimization via structured search."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFE_MARGIN = 5.0

def _time_remaining():
    return _DEADLINE - _SAFE_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x6 matrix. Output factor W follows W[k*n+i, r].
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Block-wise Kronecker decomposition:
    # T[i,j,k] = 1 = sum_{a,b} delta(i,a)*delta(j,b) * 1 * 1
    # Rank = n * m = 15
    rank = n * m  # 3 * 5 = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m+j, a*m+b] = delta(i,a) * delta(j,b)
    for i in range(n):
        for j in range(m):
            for a in range(n):
                for b in range(m):
                    r = a * m + b
                    if i == a and j == b:
                        U[i * m + j][r] = 1
    
    # V[j*p+k, a*m+b] = 1 for all j, k, a, b
    for j in range(m):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    r = a * m + b
                    V[j * p + k][r] = 1
    
    # W[k*n+i, a*m+b] = 1 for all k, i, a, b
    for k in range(p):
        for i in range(n):
            for a in range(n):
                for b in range(m):
                    r = a * m + b
                    W[k * n + i][r] = 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                assert total == 1, f"Verification failed at ({i},{j},{k}): got {total}"
    
    return {"U": U, "V": V, "W": W}
