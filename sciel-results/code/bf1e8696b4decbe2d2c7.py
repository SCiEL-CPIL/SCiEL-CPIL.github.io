"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
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

def _verify_decomposition(n, m, p, U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _search_rank_reduction(n, m, p, target_rank, time_limit=10.0):
    """Try to find a rank reduction below schoolbook using random search."""
    start = time.time()
    import random
    
    # This is a placeholder for a real search.
    # For now, we just return None to indicate no better solution found.
    # A real implementation would try to solve the system of equations.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # The schoolbook rank is 60 for 3x4x4x5
    # Let's try rank 59, 58, etc.
    
    # For now, just return the schoolbook as it's guaranteed valid
    # A proper search would require solving a system of polynomial equations
    
    return best
