"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition with rank = n*m*p."""
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
    return U, V, W

def _try_low_rank_decomposition(n, m, p, target_rank, time_limit=30.0):
    """
    Try to find a rank-reduced decomposition using structured random search.
    Uses a hybrid approach: start from schoolbook-like structure and perturb.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Strategy 1: Try structured decompositions based on known patterns
    # For 3x4 * 4x4, we can try ranks below 48
    
    # First, let's try a systematic approach: use integer entries in {-1, 0, 1}
    # and search for valid decompositions
    
    best_result = None
    
    # Number of random restarts based on time budget
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial decomposition
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Local search: try to fix violations
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Check if valid
            if _verify_decomposition(U, V, W, n, m, p, target_rank):
                return U, V, W
            
            # Find a violation and try to fix it
            # Pick a random (i, j, k) tuple
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            # Compute current value
            current = 0
            for r in range(target_rank):
                current += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            
            if current == 1:
                continue
            
            # Try modifying one of the three matrices at a random rank index
            r = random.randint(0, target_rank - 1)
            choice = random.choice(['U', 'V', 'W'])
            
            if choice == 'U':
                old_val = U[i * m + j][r]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    U[i * m + j][r] = new_val
                    # Quick check: does this help?
                    new_current = 0
                    for rr in range(target_rank):
                        new_current += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(new_current - 1) < abs(current - 1):
                        break
                    else:
                        U[i * m + j][r] = old_val
                        break
            elif choice == 'V':
                old_val = V[j * p + k][r]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    V[j * p + k][r] = new_val
                    new_current = 0
                    for rr in range(target_rank):
                        new_current += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(new_current - 1) < abs(current - 1):
                        break
                    else:
                        V[j * p + k][r] = old_val
                        break
            else:
                old_val = W[k * n + i][r]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    W[k * n + i][r] = new_val
                    new_current = 0
                    for rr in range(target_rank):
                        new_current += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(new_current - 1) < abs(current - 1):
                        break
                    else:
                        W[k * n + i][r] = old_val
                        break
    
    return None

def _find_rank_reduction_finite_field(n, m, p, target_rank, field=5, max_iters=50000, time_limit=30.0):
    """
    Try to find a rank-reduced decomposition over F_field.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize with schoolbook
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # First target_rank columns: schoolbook pattern
    for r in range(min(target_rank, n * m * p)):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Try to find dependencies to reduce rank
    # For each column r, we can try to express it as a combination of others
    # This is complex; instead, use random search with verification
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition over F_field"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(target_rank):
                        total = (total + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]) % field
                    if total != 1:
                        return False
        return True
    
    # Try random perturbations
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Randomly modify a few entries
        num_mods = random.randint(1, 5)
        for _ in range(num_mods):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                U[row][col] = random.randint(0, field - 1)
            elif mat == 'V':
                row = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                V[row][col] = random.randint(0, field - 1)
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                W[row][col] = random.randint(0, field - 1)
        
        if verify(U, V, W):
            return U, V, W
        
        # Occasionally reset to schoolbook and try a different approach
        if iteration % 1000 == 0:
            for r in range(min(target_rank, n * m * p)):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    overall_start = time.time()
    overall_deadline = overall_start + 140  # Leave 10s margin for 150s limit
    
    # Try ranks from 47 down to a reasonable minimum
    # Based on known results, the rank for 3x4*4x4 multiplication is likely around 40-45
    # Let's try from 47 down to 35, with more time for higher ranks
    
    for target_rank in range(47, 34, -1):
        if time.time() > overall_deadline - 5:
            break
        
        # Allocate more time for lower ranks (harder to find)
        time_per_rank = min(15.0, (overall_deadline - time.time()) / (47 - target_rank + 1))
        
        # First try the integer-based local search
        result = _try_low_rank_decomposition(n, m, p, target_rank, time_limit=time_per_rank / 2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Then try the finite field approach
        result = _find_rank_reduction_finite_field(n, m, p, target_rank, field=5, max_iters=10000, time_limit=time_per_rank / 2)
        if result is not None:
            U, V, W = result
            # Lift to integers (values are already 0-4, which are integers)
            return {"U": U, "V": V, "W": W}
    
    # If no reduction found, return schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
