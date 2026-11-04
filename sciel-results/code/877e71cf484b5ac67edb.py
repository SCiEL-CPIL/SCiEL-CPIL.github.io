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
    deadline = start_time + 115  # safety margin before 120s limit
    
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
    
    # The decomposition condition:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use a structured approach based on known low-rank decompositions.
    # For 3x3 x 3x3, the best known rank is 23 (Strassen-like).
    # For 3x3 x 3x8, we can try to extend or find a new decomposition.
    
    # First, let's try a deterministic approach using linear algebra.
    # We can set up the system of equations and try to solve it.
    
    # The number of unknowns for rank r is:
    # (n*m + m*p + p*n) * r = (9 + 24 + 24) * r = 57 * r
    # The number of equations is n*m*p = 72
    
    # For r >= 72/57 ≈ 1.26, we have more unknowns than equations,
    # but the equations are nonlinear (trilinear).
    
    # Let's try a different approach: use the fact that the tensor is separable
    # in a certain sense. We can try to find a decomposition where U, V, W
    # have a specific structure.
    
    # Approach 1: Try to use a known 3x3x3 rank-23 decomposition and extend it.
    # The 3x3x3 tensor T_{i,j,k} = 1 for all i,j,k in {0,1,2}.
    # A rank-23 decomposition would have 23 terms.
    # For 3x3 x 3x8, we need to handle 8 columns instead of 3.
    
    # Let's try a more systematic search with better initialization.
    
    def check_decomposition_fast(U, V, W, rank):
        """Fast check using vectorized operations."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can vectorize this.
        
        # Create indices
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # For each rank component r, compute the product
        # total[i,j,k] = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        # Check if all entries are 1
        if np.max(np.abs(total - 1.0)) > 1e-6:
            return False
        return True
    
    # Try ranks from 36 upward
    # Use a smarter initialization: start with schoolbook and try to merge terms.
    
    # Approach 2: Use a genetic algorithm or local search.
    # Start with schoolbook (rank 72) and try to reduce the rank by merging
    # terms or setting some entries to zero.
    
    # Approach 3: Use a known decomposition for smaller cases and extend.
    
    # Let's try a structured random search with better properties.
    # Use values from a larger set: {-2, -1, -0.5, 0, 0.5, 1, 2}
    
    value_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    for rank in range(36, 72):
        if time.time() > deadline - 5:
            break
        
        # Try different numbers of attempts based on rank
        # Lower rank is harder, so try more attempts
        max_attempts = max(50, 200 - (rank - 36) * 2)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with structured patterns
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Use a mixture of sparse and dense patterns
            density = random.uniform(0.3, 0.7)
            
            for i in range(n * m):
                for r in range(rank):
                    if random.random() < density:
                        U[i][r] = random.choice(value_set)
            
            for i in range(m * p):
                for r in range(rank):
                    if random.random() < density:
                        V[i][r] = random.choice(value_set)
            
            for i in range(p * n):
                for r in range(rank):
                    if random.random() < density:
                        W[i][r] = random.choice(value_set)
            
            if check_decomposition_fast(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                return best_result
        
        # Also try a structured approach based on Kronecker products or
        # block decompositions.
        if rank >= 40 and random.random() < 0.1:
            # Try a block-based decomposition
            # Divide the 8 columns into 2 blocks of 4
            # Use a 3x3 x 3x4 decomposition for each block
            
            U_block = [[0] * (rank // 2) for _ in range(n * m)]
            V_block = [[0] * (rank // 2) for _ in range(m * p // 2)]
            W_block = [[0] * (rank // 2) for _ in range((p // 2) * n)]
            
            # Try to find a rank-rank//2 decomposition for 3x3 x 3x4
            # This is still complex, so skip for now
            pass
    
    # Approach 4: Use a local search starting from schoolbook.
    # Start with rank 72 and try to reduce by merging terms.
    
    if time.time() < deadline - 10:
        # Try to reduce the rank from 72
        current_rank = 72
        U_cur = [row[:] for row in U_school]
        V_cur = [row[:] for row in V_school]
        W_cur = [row[:] for row in W_school]
        
        for _ in range(100):  # Try up to 100 reduction steps
            if time.time() > deadline - 5:
                break
            
            if current_rank <= 36:
                break
            
            # Try to remove one term (column)
            # Check if we can set one column to zero and still satisfy the equations
            # This is equivalent to checking if the remaining terms can still
            # produce the required tensor values.
            
            # For each column, check if removing it breaks the decomposition
            # If not, remove it.
            
            removed = False
            for col in range(current_rank):
                # Temporarily set this column to zero
                U_save = [row[col] for row in U_cur]
                V_save = [row[col] for row in V_cur]
                W_save = [row[col] for row in W_cur]
                
                for i in range(n * m):
                    U_cur[i][col] = 0
                for i in range(m * p):
                    V_cur[i][col] = 0
                for i in range(p * n):
                    W_cur[i][col] = 0
                
                if check_decomposition_fast(U_cur, V_cur, W_cur, current_rank):
                    # This column can be removed
                    # Remove it from all matrices
                    for i in range(n * m):
                        U_cur[i].pop(col)
                    for i in range(m * p):
                        V_cur[i].pop(col)
                    for i in range(p * n):
                        W_cur[i].pop(col)
                    current_rank -= 1
                    removed = True
                    break
                else:
                    # Restore
                    for i in range(n * m):
                        U_cur[i][col] = U_save[i]
                    for i in range(m * p):
                        V_cur[i][col] = V_save[i]
                    for i in range(p * n):
                        W_cur[i][col] = W_save[i]
            
            if not removed:
                break
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_result = {
                "U": U_cur,
                "V": V_cur,
                "W": W_cur
            }
    
    return best_result
# EVOLVE-BLOCK-END
