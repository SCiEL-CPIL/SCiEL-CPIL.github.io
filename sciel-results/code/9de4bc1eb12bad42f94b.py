"""Rewritten program: attempt to find a lower-rank exact tensor decomposition for 2x3 by 3x7."""
import time
import random
import math
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] == 1 if (i,j,k) is valid else 0
    Actually the tensor T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k? 
    No wait. Let me re-read the contract.
    
    The decomposition is: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where T_{i,j,k} = 1 for the matrix multiplication tensor.
    Actually for matrix multiplication, T_{i,j,k} = 1 if we're computing (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor is T_{i,j,k} = 1 for all valid i,j,k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0]) if best["U"] else 0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Try to find a lower rank decomposition using random search + local optimization
    # We'll try ranks from lower values up
    
    # Known lower bounds: For m x n matrix multiplication, rank >= max(m,n) * something
    # For 2x3 * 3x7, let's try various ranks
    
    # Strategy: Random initialization with structured patterns, then verify
    # Also try known constructions
    
    # First, let's try a simple structured approach
    # For small matrices, we can try all possible patterns
    
    # Let's try rank 36, 35, ... down to some minimum
    # Minimum possible rank: at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # But that's a very loose bound. The actual minimum is likely higher.
    
    # Let's use a randomized approach:
    # 1. Initialize U, V, W with random values from a small set {-1, 0, 1}
    # 2. Check if the tensor equation is satisfied
    # 3. If not, try to fix by adjusting entries
    
    # Actually, let's try a more systematic approach:
    # The tensor equation is: for all (i,j,k), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is a system of n*m*p = 42 equations in rank^3 variables (but structured).
    
    # Let's try random search with verification
    rng = random.Random(42)
    
    def try_random_rank(rank, max_iters=1000):
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            # Random initialization
            U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(nm)]
            V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(mp)]
            W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(pn)]
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if s != 1:
                            valid = False
                            break
            if valid:
                return {"U": U, "V": V, "W": W}
        return None
    
    # Try decreasing ranks
    for rank in range(40, 20, -1):
        if time.time() > deadline:
            break
        result = try_random_rank(rank, max_iters=500)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # If we found something better, use it; otherwise use schoolbook
    if best_rank < 42:
        return best
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
