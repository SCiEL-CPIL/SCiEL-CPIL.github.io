"""Combined program for matrix multiplication 3,4,5 tensor rank minimization.
Uses schoolbook as baseline and attempts integer-based search for lower ranks."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 110  # Leave 10s margin for the 120s limit
    
    # 1. Schoolbook Decomposition (Rank 60)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=int)
    V_school = np.zeros((mp, rank_school), dtype=int)
    W_school = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 2  # Scaled by 2
                V_school[j*p + k, col] = 2  # Scaled by 2
                W_school[k*n + i, col] = 2  # Scaled by 2
                col += 1
    
    # Verify schoolbook: Sum(2*2*2) = 8 for each (i,j,k). Correct.
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # 2. Search for lower rank decompositions
    # We work with scaled integers where the target sum is 8.
    # U, V, W entries are integers.
    # Target: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 8
    
    def verify_scaled(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if s != 8:
                        return False
        return True

    random.seed(42)
    
    # Try ranks from 59 down to 50
    # Using sparse random integers (-2, -1, 0, 1, 2) to keep magnitudes small
    for r in range(rank_school - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Number of trials decreases as rank decreases (harder to find)
        num_trials = 100 if r > 55 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers
            # Bias towards sparsity and small values
            U = np.random.choice([-2, -1, 0, 1, 2], size=(nm, r))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(mp, r))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(pn, r))
            
            # Quick check: If any row is all zeros, it's likely invalid, but verify handles it.
            # We can add a heuristic: ensure each row has at least one non-zero?
            # For now, just verify.
            
            if verify_scaled(U, V, W):
                # Found a valid decomposition!
                best_rank = r
                best_U, best_V, best_W = U, V, W
                # Convert back to half-integers: divide by 2
                best_U = best_U / 2.0
                best_V = best_V / 2.0
                best_W = best_W / 2.0
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }

    # If no lower rank found, return the schoolbook decomposition (scaled back)
    # Schoolbook was built with 2s, so dividing by 2 gives 1s.
    U_out = best_U / 2.0
    V_out = best_V / 2.0
    W_out = best_W / 2.0
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END
