"""Matrix multiplication 2,4,7 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i,j,k (the all-ones tensor representing
    that every term A[i,j]*B[j,k] contributes to C[i,k]).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_low_rank_decomposition(n, m, p, max_rank, time_budget):
    """Try to find a decomposition with rank <= max_rank using structured search.
    
    Strategy:
    1. Try to find a decomposition by exploiting the tensor structure
    2. Use random initialization with local optimization
    3. Try to merge components that have overlapping support
    """
    start_time = time.time()
    
    # First, try a structured approach based on the tensor structure
    # The key insight: T[i,j,k] = 1 for all i,j,k
    # We can try to find a decomposition with fewer than n*m*p terms
    
    # Approach 1: Try to find a rank-r decomposition where r < n*m*p
    # by using the fact that the tensor has a specific structure
    
    # For 2x4x7, let's try ranks from 55 down
    for r in range(min(max_rank, 55), 10, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Try random initialization with local search
        best_result = _random_search_with_local_opt(n, m, p, r, time_budget - (time.time() - start_time))
        if best_result is not None:
            return best_result
    
    return None

def _random_search_with_local_opt(n, m, p, r, time_budget):
    """Try to find a rank-r decomposition using random search with local optimization."""
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < time_budget and attempts < 50:
        attempts += 1
        
        # Random initialization
        U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(p * n)]
        
        # Local optimization: try to fix errors by flipping signs or adding/removing 1s
        improved = True
        max_iter = 100
        for _ in range(max_iter):
            if time.time() - start_time > time_budget:
                break
            
            improved = False
            # Check each tensor entry
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                        if abs(current - 1) > 1e-9:
                            # Try to fix by modifying one of the entries
                            fixed = False
                            for c in range(r):
                                # Try flipping U
                                old_u = U[i*m+j][c]
                                U[i*m+j][c] = -old_u
                                new_current = sum(U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2] for c2 in range(r))
                                if abs(new_current - 1) < abs(current - 1):
                                    improved = True
                                    fixed = True
                                    break
                                else:
                                    U[i*m+j][c] = old_u
                            
                            if not fixed:
                                # Try flipping V
                                for c in range(r):
                                    old_v = V[j*p+k][c]
                                    V[j*p+k][c] = -old_v
                                    new_current = sum(U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2] for c2 in range(r))
                                    if abs(new_current - 1) < abs(current - 1):
                                        improved = True
                                        fixed = True
                                        break
                                    else:
                                        V[j*p+k][c] = old_v
                            
                            if not fixed:
                                # Try flipping W
                                for c in range(r):
                                    old_w = W[k*n+i][c]
                                    W[k*n+i][c] = -old_w
                                    new_current = sum(U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2] for c2 in range(r))
                                    if abs(new_current - 1) < abs(current - 1):
                                        improved = True
                                        fixed = True
                                        break
                                    else:
                                        W[k*n+i][c] = old_w
                            
                            if not fixed:
                                # Try setting a zero to 1 or 1 to 0
                                for c in range(r):
                                    if U[i*m+j][c] == 0 and V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                                        U[i*m+j][c] = 1
                                        improved = True
                                        fixed = True
                                        break
                                    elif U[i*m+j][c] == 1 and V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                                        U[i*m+j][c] = 0
                                        improved = True
                                        fixed = True
                                        break
                            
                            if fixed:
                                break
            
            if not improved:
                break
        
        # Verify the result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Start with the schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try to improve using random search with local optimization
    time_remaining = deadline - (time.time() - start_time)
    if time_remaining > 5:
        result = _try_low_rank_decomposition(n, m, p, best_rank, time_remaining)
        if result is not None and len(result["U"][0]) < best_rank:
            best = result
            best_rank = len(best["U"][0])
    
    return best

# EVOLVE-BLOCK-END
