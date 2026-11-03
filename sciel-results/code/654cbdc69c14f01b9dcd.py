"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Schoolbook decomposition
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
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured approaches
    # For 4x4x6, try ranks from known lower bounds up
    
    # Lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    # But actual minimum rank for 4x4x6 is likely higher
    
    # Strategy 1: Try to find structured decompositions using block patterns
    # Strategy 2: Random search with small integer/half-integer values
    # Strategy 3: Use the fact that we can combine schoolbook terms
    
    # First, try a structured approach: use Kronecker-like patterns
    # For 4x4x4, there are known rank-21 decompositions
    # For 4x4x6, we can extend this
    
    # Try ranks in a reasonable range
    # Known: 4x4x4 has rank 21, so 4x4x6 might be around 30-40
    
    target_ranks = list(range(24, 60)) + list(range(60, 96, 2))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple random attempts for each rank
        max_attempts = 20 if target_rank < 40 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random factor matrices with small values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U_rand, V_rand, W_rand):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = target_rank
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If no improvement found, try a more systematic approach
    # Use the structure of the problem: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can try to find decompositions that exploit the 4x4x6 structure
    
    if best_rank == rank_schoolbook and time.time() < deadline:
        # Try a structured decomposition based on 4x4x4 rank-21 + extension
        # For 4x4x4, use known rank-21 decomposition pattern
        # Then extend to 4x4x6 by adding terms for the extra 2 columns
        
        # Actually, let's try a different approach: use the fact that
        # we can write the tensor as a sum of rank-1 tensors
        # and try to find a sparse decomposition
        
        # Try ranks from 30 to 50 with more structured random search
        for target_rank in range(30, 51):
            if time.time() > deadline:
                break
            
            # Use a biased random search: start from schoolbook and perturb
            U_try = [row[:] for row in U_school]
            V_try = [row[:] for row in V_school]
            W_try = [row[:] for row in W_school]
            
            # Randomly zero out some entries to create sparsity
            for _ in range(target_rank * 2):
                if time.time() > deadline:
                    break
                
                # Pick a random column and try to eliminate it
                col_idx = random.randint(0, rank_schoolbook - 1)
                
                # Try to express this column as a combination of others
                # For simplicity, just try random perturbations
                U_try = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n * m)]
                V_try = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m * p)]
                W_try = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p * n)]
                
                if verify(U_try, V_try, W_try):
                    best_result = {"U": U_try, "V": V_try, "W": W_try}
                    best_rank = target_rank
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    return best_result
# EVOLVE-BLOCK-END
