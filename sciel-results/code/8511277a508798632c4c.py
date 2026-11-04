"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Precompute index arrays for fast evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition_fast(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        if np.max(np.abs(total - 1.0)) > 1e-6:
            return False
        return True

    def check_exact_integer(U, V, W, rank):
        """Check if integer decomposition is exact"""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        if np.max(np.abs(total - 1.0)) > 1e-9:
            return False
        return True

    # Strategy: Tensor flattening approach
    # Treat the multiplication tensor as a linear operator and find rank-1 components
    # For 3x3x3x8, we can try to find a decomposition with rank < 72
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8)
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r] where:
    #   U has shape (n*m, rank)
    #   V has shape (m*p, rank)  
    #   W has shape (p*n, rank)
    
    # Try ranks from 50 to 71
    for rank_target in range(50, min(rank_school, 72)):
        if time.time() > deadline - 20:
            break
            
        success = False
        
        # Multiple random restarts for each rank
        max_restarts = 5 if rank_target > 60 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random sparse pattern
            U_arr = np.zeros((n * m, rank_target), dtype=np.float64)
            V_arr = np.zeros((m * p, rank_target), dtype=np.float64)
            W_arr = np.zeros((p * n, rank_target), dtype=np.float64)
            
            # Random initialization with values from {-1, 0, 1}
            for r in range(rank_target):
                # U: each column gets random ±1 for some rows
                num_nonzero_u = random.randint(3, 9)
                rows_u = random.sample(range(n * m), num_nonzero_u)
                for row in rows_u:
                    U_arr[row, r] = random.choice([-1, 1])
                
                # V: each column gets random ±1 for some rows
                num_nonzero_v = random.randint(3, 12)
                rows_v = random.sample(range(m * p), num_nonzero_v)
                for row in rows_v:
                    V_arr[row, r] = random.choice([-1, 1])
                
                # W: each column gets random ±1 for some rows
                num_nonzero_w = random.randint(3, 9)
                rows_w = random.sample(range(p * n), num_nonzero_w)
                for row in rows_w:
                    W_arr[row, r] = random.choice([-1, 1])
            
            # Local search: perturb coefficients to reduce error
            best_err = np.inf
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current error
                total = np.zeros(n * m * p, dtype=np.float64)
                for r in range(rank_target):
                    total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                err = np.sum((total - 1.0)**2)
                
                if err < 1e-9:
                    # Check for exact integer solution
                    U_int = [[int(round(x)) for x in row] for row in U_arr.tolist()]
                    V_int = [[int(round(x)) for x in row] for row in V_arr.tolist()]
                    W_int = [[int(round(x)) for x in row] for row in W_arr.tolist()]
                    if check_exact_integer(U_int, V_int, W_int, rank_target):
                        best_rank = rank_target
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        success = True
                        break
                    
                if err < best_err:
                    best_err = err
                
                # Pick a random element to perturb
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, rank_target - 1)
                    U_arr[row, col] += random.choice([-1, 1])
                elif mat_idx == 1:
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, rank_target - 1)
                    V_arr[row, col] += random.choice([-1, 1])
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, rank_target - 1)
                    W_arr[row, col] += random.choice([-1, 1])
            
            if success:
                break
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
