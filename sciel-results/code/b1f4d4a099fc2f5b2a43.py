"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify exact integer decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _verify_decomposition_mod(U, V, W, n, m, p, rank, field):
    """Verify decomposition over F_field."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total = (total + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]) % field
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

def _search_low_rank(n, m, p, target_rank, field, time_limit=5.0):
    """
    Search for a rank-reduced decomposition over F_field.
    Uses hill-climbing with constraint satisfaction.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Initialize with schoolbook pattern for first min(target_rank, 48) columns
    num_init = min(target_rank, n * m * p)
    for r in range(num_init):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    def count_satisfied():
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(target_rank):
                        total = (total + U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]) % field
                    if total == 1:
                        count += 1
        return count
    
    def is_valid():
        return count_satisfied() == n * m * p
    
    if is_valid():
        return U, V, W
    
    best_count = count_satisfied()
    total_constraints = n * m * p
    
    # Hill-climbing with occasional random restarts
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        if is_valid():
            return U, V, W
        
        # Modify a random entry
        mat_choice = random.randint(0, 2)
        if mat_choice == 0:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, target_rank - 1)
            old_val = U[row][col]
            U[row][col] = random.randint(0, field - 1)
        elif mat_choice == 1:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, target_rank - 1)
            old_val = V[row][col]
            V[row][col] = random.randint(0, field - 1)
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, target_rank - 1)
            old_val = W[row][col]
            W[row][col] = random.randint(0, field - 1)
        
        new_count = count_satisfied()
        if new_count >= best_count:
            best_count = new_count
        else:
            # Revert if we got worse
            if mat_choice == 0:
                U[row][col] = old_val
            elif mat_choice == 1:
                V[row][col] = old_val
            else:
                W[row][col] = old_val
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    start_time = time.time()
    global_deadline = start_time + 120  # 120s total limit
    
    # Try to find lower rank decompositions
    # Focus on finite fields where computation is fast
    fields_to_try = [2, 3, 5]
    
    # Try ranks from high to low, but with limited time per attempt
    # Be strategic: try a few ranks per field
    for field in fields_to_try:
        if time.time() > global_deadline - 10:
            break
        
        # Try ranks: focus on ranges where reduction is plausible
        # Start from 47 and go down, but limit total attempts
        ranks_to_try = list(range(47, 25, -1))
        
        for target_rank in ranks_to_try:
            if time.time() > global_deadline - 5:
                break
            
            time_per_attempt = min(3.0, (global_deadline - 5 - time.time()) / max(1, len(ranks_to_try) - ranks_to_try.index(target_rank)))
            
            result = _search_low_rank(n, m, p, target_rank, field, time_limit=time_per_attempt)
            if result is not None:
                U, V, W = result
                # Verify over integers
                if _verify_decomposition(U, V, W, n, m, p, target_rank):
                    return {"U": U, "V": V, "W": W}
    
    # If no reduction found, return schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
