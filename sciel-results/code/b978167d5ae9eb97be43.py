"""Program attempting to find a rank-minimal decomposition for 2x3x9 matrix multiplication."""
import time
import sys
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120  # Keep a safety margin from the 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Baseline: Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                U_sb[i * m + j][idx] = 1
                V_sb[j * p + k][idx] = 1
                W_sb[k * n + i][idx] = 1
                
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Values to search from
    values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    values_array = np.array(values)
    
    # Helper to check validity using NumPy einsum
    def check_valid(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_res = U.reshape(n, m, R)
        V_res = V.reshape(m, p, R)
        W_res = W.reshape(p, n, R)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by k, i in the formula. 
        # W_res[k, i, r] corresponds to W[k*n + i, r].
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_res, V_res, W_res)
        except Exception:
            return False
            
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bound for rank of matrix mult is max(n, m, p) is not quite right, 
    # but for 2x3x9, rank 54 is schoolbook. 
    # Let's try down to 10.
    
    for target_rank in range(10, min(rank_sb + 1, 40)):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts scales with remaining time and rank complexity
        # Higher ranks are easier to satisfy but we want minimal rank.
        # We do more attempts for lower ranks.
        attempts = 2000 if target_rank < 20 else 500
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = values_array[random.randint(0, len(values)-1)] # This is wrong, need arrays
            # Correct random init:
            U = np.random.choice(values_array, size=(n * m, target_rank))
            V = np.random.choice(values_array, size=(m * p, target_rank))
            W = np.random.choice(values_array, size=(p * n, target_rank))
            
            if check_valid(U, V, W, target_rank):
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                return best
        
    # If no better decomposition found, return the schoolbook baseline
    return best
# EVOLVE-BLOCK-END
