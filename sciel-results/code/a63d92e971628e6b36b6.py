"""Rewritten program for 3x5x6 matrix multiplication rank search."""
DIMENSIONS = [3, 5, 6]

import sys
import time
import random
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid matrix multiplication decomposition.
    
    Condition: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _try_rank_r(n, m, p, r, deadline, seed=None):
    """Try to find a rank-r decomposition using randomized approach with rational entries."""
    if seed is not None:
        random.seed(seed)
    
    def rand_rational(max_num=4, max_den=4):
        num = random.randint(-max_num, max_num)
        den = random.randint(1, max_den)
        return Fraction(num, den)
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Initialize with small rationals
        for i in range(n*m):
            for c in range(r):
                U[i][c] = rand_rational()
        for j in range(m*p):
            for c in range(r):
                V[j][c] = rand_rational()
        for k in range(p*n):
            for c in range(r):
                W[k][c] = rand_rational()
        
        if _verify_decomposition(U, V, W, n, m, p, r, tol=1e-12):
            return {"U": U, "V": V, "W": W}
    
    return None

def _construct():
    """Main construction function."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound for rank: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from a reasonable upper bound down to lower bound
    # Focus on ranks that are achievable
    candidate_ranks = []
    for r in range(60, lower_bound - 1, -1):
        candidate_ranks.append(r)
    
    # Try each candidate rank
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
        
        result = _try_rank_r(n, m, p, r, deadline, seed=r*1000 + 42)
        if result is not None:
            best = result
            best_rank = r
            # If we found a decomposition, continue trying lower ranks
            # but with less time per rank
            if r <= lower_bound + 5:
                break
    
    return best

def construct():
    return _construct()

# EVOLVE-BLOCK-END
