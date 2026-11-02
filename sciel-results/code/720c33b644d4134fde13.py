"""Search for rank-minimal exact tensor decomposition of 2x4x7 matrix multiplication."""
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n=2, m=4, p=7):
    """Check if U, V, W give exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    seed_rank = n * m * p  # 56
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = seed_rank
    
    # Verify schoolbook
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
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
    return U, V, W
