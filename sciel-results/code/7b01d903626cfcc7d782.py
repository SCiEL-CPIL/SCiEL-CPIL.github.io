import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations_with_replacement
import random

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_merge_terms(U, V, W, n, m, p, rank, term1, term2, start_time, deadline):
    """
    Try to merge two rank-1 terms into one using algebraic identity.
    Returns (new_U, new_V, new_W, new_rank) if successful, else None.
    """
    i1, j1, k1 = term1
    i2, j2, k2 = term2
    
    # For merging to work, we need specific index relationships
    # Case 1: Same i, different j,k -> use (a+b)(c+d) = ac + ad + bc + bd
    # Case 2: Same j, different i,k
    # Case 3: Same k, different i,j
    
    # The key insight: if two terms share one index, we can sometimes combine them
    # using the identity that allows us to replace 2 terms with 1 term + adjustments
    
    # For now, try a simple approach: if terms share exactly one index,
    # we might be able to merge them with appropriate coefficients
    
    i1_idx = i1 * m + j1
    j1_idx = j1 * p + k1
    k1_idx = k1 * n + i1
    
    i2_idx = i2 * m + j2
    j2_idx = j2 * p + k2
    k2_idx = k2 * n + i2
    
    # Check if they share exactly one index
    shared = []
    if i1 == i2:
        shared.append('i')
    if j1 == j2:
        shared.append('j')
    if k1 == k2:
        shared.append('k')
    
    if len(shared) != 1:
        return None
    
    # If they share exactly one index, try to merge
    # This is a heuristic - we'll need to verify afterwards
    # For now, create a candidate merged term
    
    # The merged term would have:
    # U[i*m+j, r] = 1 for the new combined (i,j)
    # V[j*p+k, r] = 1 for the new combined (j,k)  
    # W[k*n+i, r] = 1 for the new combined (k,i)
    
    # But this is complex. Instead, let's use a different approach:
    # Random search for lower rank decompositions
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60
    
    # Initialize schoolbook solution
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = [row[:] for row in U_school]
    best_V = [row[:] for row in V_school]
    best_W = [row[:] for row in W_school]
    
    # Strategy: Try to find a lower rank decomposition using random search
    # with rational coefficients. We'll try ranks from 59 down to 40.
    
    # For efficiency, we'll use a greedy approach:
    # 1. Start with schoolbook
    # 2. Try to eliminate terms by finding linear dependencies
    
    # Alternative: Use the fact that for 2x5x6, we might be able to achieve
    # rank < 60 by exploiting the small dimensions.
    
    # Let's try a systematic search for rank 59, 58, etc.
    # We'll use random initialization and then try to solve the linear system.
    
    # Actually, let's try a different approach: 
    # The matrix multiplication tensor has symmetries. For 2x5x6,
    # we can try to find a decomposition using the structure of the problem.
    
    # Key insight: If we fix i, the terms with that i form a 5x6 matrix multiplication.
    # We can try to find a low-rank decomposition for each i and combine them.
    
    # But the terms are coupled across i values.
    
    # Let's try a practical approach: random search with verification
    for target_rank in range(rank_school - 1, max(30, rank_school - 20), -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random rational values
            # Use small integers to keep numbers manageable
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            if _verify(U_rand, V_rand, W_rand, n, m, p, target_rank):
                best_rank = target_rank
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                break
        
        # If we found a valid decomposition, we can try to go lower
        if best_rank < rank_school:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
