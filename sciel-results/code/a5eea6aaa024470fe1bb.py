"""Program attempting to find a rank-minimal decomposition for 2x3x9 matrix multiplication."""
import time
import sys
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # Keep a safety margin from the 150s limit
    
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
    
    # Helper to check validity using NumPy einsum
    def check_valid(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_res = U.reshape(n, m, R)
        V_res = V.reshape(m, p, R)
        W_res = W.reshape(p, n, R)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_res, V_res, W_res)
        except Exception:
            return False
            
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Try to find a better decomposition using structured search
    # For 2x3x9, let's try ranks from a reasonable lower bound upward
    # Theoretical lower bound is at least max(n*m, m*p, p*n) / something, but let's be practical
    
    # Strategy: Start from schoolbook and try to eliminate terms
    # Or try specific patterns
    
    # First, try a few structured patterns
    # Pattern 1: Try to use the fact that n=2 is small
    
    # Let's try ranks from 30 to 54 (schoolbook)
    # We'll use a more intelligent search
    
    for target_rank in range(30, rank_sb + 1):
        if time.time() > deadline - 5:
            break
        
        # For each target rank, try multiple random initializations
        # But with a smarter approach: start from a valid decomposition and perturb
        
        if target_rank == rank_sb:
            # This is the schoolbook, we already have it
            continue
            
        # Try random search with smaller value set for speed
        values_small = [-1, 0, 1]
        values_array = np.array(values_small)
        
        attempts = 1000 if target_rank < 40 else 200
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small values
            U = np.random.choice(values_array, size=(n * m, target_rank))
            V = np.random.choice(values_array, size=(m * p, target_rank))
            W = np.random.choice(values_array, size=(p * n, target_rank))
            
            if check_valid(U, V, W, target_rank):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                return best
        
        # If we haven't found anything better, try a more targeted approach
        # for lower ranks: use the structure of the problem
        
        if target_rank <= 35:
            # Try to construct a decomposition based on block structure
            # For 2x3x9, we can think of it as multiplying [2x3] by [3x9]
            
            # One approach: use the fact that we only need 2*3*9 = 54 products
            # but maybe we can share some computations
            
            # Let's try a systematic search with more structured initialization
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Initialize with a pattern: some entries are 1, others 0
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # Randomly assign 1s to some positions
                for r in range(target_rank):
                    # Each rank-1 term should contribute to some subset of the tensor
                    # Randomly choose which (i,j), (j,k), (k,i) this term affects
                    
                    # For U: choose which (i,j) pairs get a 1
                    num_u = random.randint(1, min(6, target_rank))
                    u_indices = random.sample(range(n * m), num_u)
                    for idx in u_indices:
                        U[idx, r] = 1
                    
                    # For V: choose which (j,k) pairs get a 1
                    num_v = random.randint(1, min(9, target_rank))
                    v_indices = random.sample(range(m * p), num_v)
                    for idx in v_indices:
                        V[idx, r] = 1
                    
                    # For W: choose which (k,i) pairs get a 1
                    num_w = random.randint(1, min(9, target_rank))
                    w_indices = random.sample(range(p * n), num_w)
                    for idx in w_indices:
                        W[idx, r] = 1
                
                if check_valid(U, V, W, target_rank):
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    return best
    
    # If no better decomposition found, return the schoolbook baseline
    return best
# EVOLVE-BLOCK-END
