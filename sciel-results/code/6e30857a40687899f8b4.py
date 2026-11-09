"""Program that attempts to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]
SEED_RANK = 105

# EVOLVE-BLOCK-START

def _validate(U, V, W, n=3, m=5, p=7):
    """
    Validate that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The condition is: for all i, j, k,
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n=3, m=5, p=7):
    """Return the schoolbook decomposition with rank n*m*p."""
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


def _attempt_rank_reduction(n=3, m=5, p=7, deadline=None, seed=42):
    """
    Attempt to find a lower-rank decomposition using structured approaches.
    For small dimensions like 3x5x7, we use a random search over 
    integer/half-integer coefficients with a structured initialization.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    rng = random.Random(seed)
    best = _schoolbook(n, m, p)
    
    # The tensor we need to decompose: T[i,j,k] = 1 for all i,j,k
    # A rank-r decomposition gives:
    # T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def validate2(U, V, W, n=3, m=5, p=7):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Try ranks from 104 down to some minimum
    for target_rank in range(SEED_RANK - 1, max(10, SEED_RANK // 2)):
        if time.time() > deadline:
            break
        
        # Try random structured approaches
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers/half-integers
            U = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if validate2(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return best


def construct():
    """
    Returns the best tensor decomposition found for matrix multiplication.
    """
    deadline = time.time() + 140
    
    # First, return schoolbook as a safe fallback
    best = _schoolbook()
    
    # Attempt rank reduction
    if time.time() < deadline:
        result = _attempt_rank_reduction(deadline=deadline)
        if result is not None:
            best = result
    
    return best

# EVOLVE-BLOCK-END
