"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    def validate(U, V, W, rank):
        # Check if U, V, W form a valid decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c_idx in range(p*n):
                    # Skip the non-zero entries
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    i2 = c_idx // n
                    k2 = c_idx % n
                    if i == i2 and j == j2 and k == k2:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Search for lower ranks
    for target_rank in range(16, rank_school):
        if time.time() > deadline:
            break
        
        # Try random initialization with half-integers
        found = False
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + target_rank)
            # Generate random half-integer matrices
            def rand_half_int():
                vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
                return random.choice(vals)
            
            U = [[rand_half_int() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(target_rank)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(target_rank)] for _ in range(p*n)]
            
            if validate(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we can't find rank target_rank, try higher
            continue
    
    return best_result
# EVOLVE-BLOCK-END
