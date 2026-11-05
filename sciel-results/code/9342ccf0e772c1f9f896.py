"""Optimized program for 3x3 * 3x5 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = 140  # seconds - safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using randomized search
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For each candidate rank, try to find a valid decomposition
    # We'll use a simple approach: try ranks from low to high
    
    # Lower bound considerations:
    # For 3x3x3, rank is 23
    # For 3x5x3, we expect rank to be lower than schoolbook (45)
    # Let's try ranks from 15 to 44
    
    for target_rank in range(15, rank_schoolbook):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Random initialization with rational numbers (integers or half-integers)
            # Use small integers to keep values manageable
            U = np.random.randint(-3, 4, size=(n*m, target_rank))
            V = np.random.randint(-3, 4, size=(m*p, target_rank))
            W = np.random.randint(-3, 4, size=(p*n, target_rank))
            
            # Check if this decomposition is exact
            # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum over c
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid decomposition
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank == target_rank:
            break  # Found the minimal rank
    
    return best_result

# EVOLVE-BLOCK-END
