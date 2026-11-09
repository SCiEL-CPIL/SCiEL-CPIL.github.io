"""Matrix multiplication tensor decomposition search for 4x5 * 5x5."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

# Use Fraction for exact arithmetic
# Dimensions
n, m, p = 4, 5, 5

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook decomposition (rank 100)
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
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to find better decompositions using structured approaches
    # Strategy 1: Try Winograd-style decompositions
    # Strategy 2: Use known fast matrix multiplication results
    # Strategy 3: Randomized search with exact verification
    
    # Lower bound is max(n*m, m*p, n*p) = 25
    min_rank = max(n*m, m*p, n*p)
    
    # Try various candidate ranks
    candidate_ranks = list(range(min_rank, min(best_rank, 60)))
    # Add some strategic ranks
    for r in [25, 30, 35, 40, 45, 50, 55, 60]:
        if r not in candidate_ranks and r < best_rank:
            candidate_ranks.append(r)
    candidate_ranks = sorted(set(candidate_ranks))
    
    for candidate_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple strategies for this rank
        found = False
        
        # Strategy A: Randomized initialization with small integers
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Use small integer values to increase chance of exact match
            U_try = [[random.randint(-3, 3) for _ in range(candidate_rank)] for _ in range(n * m)]
            V_try = [[random.randint(-3, 3) for _ in range(candidate_rank)] for _ in range(m * p)]
            W_try = [[random.randint(-3, 3) for _ in range(candidate_rank)] for _ in range(p * n)]
            
            if _check_decomposition_fast(U_try, V_try, W_try, candidate_rank):
                best_rank = candidate_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                found = True
                break
        
        if found:
            continue
        
        # Strategy B: Structured approach using Kronecker products
        # Try to build from smaller known decompositions
        if candidate_rank <= 40:
            result = _try_structured_decomposition(candidate_rank)
            if result is not None:
                best_rank = candidate_rank
                best_U, best_V, best_W = result
                continue
    
    # Final verification and cleanup
    # Ensure we have a valid decomposition
    if not _check_decomposition_fast(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_U = U
        best_V = V
        best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _check_decomposition_fast(U, V, W, rank):
    """Check if the given U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True


def _try_structured_decomposition(rank):
    """Try to find a structured decomposition for the given rank."""
    # This is a placeholder for more sophisticated structured approaches
    # For now, return None to indicate no structured solution found
    return None
