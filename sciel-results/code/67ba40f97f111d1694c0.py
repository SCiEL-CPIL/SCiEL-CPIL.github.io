import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dict of (i,j,k) -> 1"""
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[(i, j, k)] = 1
    return tensor

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact"""
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

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: for each candidate rank r < best_rank, try to find a decomposition
    # Use randomized search over integers with small entries
    
    # The tensor has 105 ones. For rank r, we need 71r variables and 105 constraints.
    # For r >= 2, we have more variables than constraints (71*2=142 > 105), so solutions may exist.
    # But the system is nonlinear (products of variables), so it's hard.
    
    # Alternative: try structured decompositions
    # For 3x5 x 5x7, maybe we can exploit the fact that 3,5,7 are primes
    
    # Let's try a greedy peeling approach:
    # 1. Start with the full tensor (all 105 terms)
    # 2. Try to replace pairs of terms with a single term if possible
    
    # Actually, let's try a different approach: use the fact that for small matrices,
    # there might be known optimal algorithms.
    
    # For now, let's try a randomized search for rank 2 to 50
    # This is unlikely to find a solution quickly, but let's try
    
    # More practical: try to find if rank 104 or lower works by random perturbation
    # of the schoolbook and checking for linear dependencies
    
    # Actually, the most promising approach: try to find a decomposition where
    # some of the U, V, W entries are fractional, allowing more combinations.
    
    # Let's try a simple randomized search for small ranks
    import random
    
    for target_rank in range(2, min(best_rank, 50)):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random integers
            U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # If no better found, return schoolbook
    return best
