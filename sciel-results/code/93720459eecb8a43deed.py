"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook as fallback (rank 72)
    rank_school = n * m * p  # 72
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
    
    # Precompute index arrays for fast vectorized error computation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition_exact(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication using integer arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    def compute_error_vectorized(U_arr, V_arr, W_arr, rank):
        """Compute sum of squared errors using vectorized ops."""
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        return np.sum((total - 1.0)**2)
    
    # Strategy: Block decomposition of 3x3 x 3x8 into three 3x3x3 blocks and one 3x3x2 block
    # Use structured initialization based on known tensor decomposition patterns
    # For 3x3x3: use rank 9 schoolbook initially, then try to reduce
    # For 3x3x2: use rank 6 schoolbook
    
    # Block structure:
    # Block 1: k in [0, 2] - 3x3x3 multiplication
    # Block 2: k in [3, 5] - 3x3x3 multiplication  
    # Block 3: k in [6, 7] - 3x3x2 multiplication
    
    # Initialize with schoolbook for each block, then try local search to reduce rank
    # Target: reduce from 72 to something lower
    
    # First, let's try a structured approach:
    # For each 3x3x3 block, use the standard rank-9 decomposition
    # For the 3x3x2 block, use the standard rank-6 decomposition
    # Then try to merge terms across blocks using linear dependencies
    
    # Actually, let's use a more direct approach:
    # Start with schoolbook and try to find linear dependencies to reduce rank
    
    # Precompute the tensor as a 3D array
    # T[i,j,k] = 1 if this term exists in the decomposition
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # We want to find U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Approach: Use random initialization with block structure and local search
    # Key insight: The tensor is highly structured, so we can exploit this
    
    # Let's try different target ranks and use a smarter initialization
    
    best_found = False
    
    # Try target ranks from 70 down to 50
    for target_rank in range(70, 49, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 10 if target_rank <= 65 else 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with block structure
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Block sizes
            # Block 1: k=0,1,2 (3x3x3)
            # Block 2: k=3,4,5 (3x3x3)
            # Block 3: k=6,7 (3x3x2)
            
            # Allocate terms to blocks
            # For 3x3x3: need at least 9 terms for schoolbook
            # For 3x3x2: need at least 6 terms for schoolbook
            # Total minimum: 9 + 9 + 6 = 24
            
            if target_rank >= 24:
                block1_size = min(9, max(7, target_rank // 3))
                block2_size = min(9, max(7, (target_rank - block1_size) // 2))
                block3_size = target_rank - block1_size - block2_size
                
                # Ensure block3 has at least 6 terms
                if block3_size < 6:
                    block1_size = max(7, (target_rank - 6) // 2)
                    block2_size = max(7, target_rank - 6 - block1_size)
                    block3_size = target_rank - block1_size - block2_size
                
                # Initialize Block 1 (k=0,1,2) with schoolbook pattern
                for r in range(block1_size):
                    i = r % 3
                    j = (r // 3) % 3
                    k = r // 9
                    if k < 3:
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                
                # Initialize Block 2 (k=3,4,5) with schoolbook pattern
                for r in range(block1_size, block1_size + block2_size):
                    idx = r - block1_size
                    i = idx % 3
                    j = (idx // 3) % 3
                    k = 3 + idx // 9
                    if k < 6:
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                
                # Initialize Block 3 (k=6,7) with schoolbook pattern
                for r in range(block1_size + block2_size, target_rank):
                    idx = r - (block1_size + block2_size)
                    if idx < 6:
                        i = idx % 3
                        j = (idx // 3) % 3
                        k = 6 + idx // 6
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
            
            # Add some random perturbation to break symmetry
            if random.random() < 0.3:
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            if random.random() < 0.1:
                                U[i * m + j][r] += random.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if random.random() < 0.1:
                                V[j * p + k][r] += random.choice([-1, 1])
                    for k in range(p):
                        for i in range(n):
                            if random.random() < 0.1:
                                W[k * n + i][r] += random.choice([-1, 1])
            
            # Local search with gradient-like updates
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            best_err = compute_error_vectorized(U_arr, V_arr, W_arr, target_rank)
            
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                err = compute_error_vectorized(U_arr, V_arr, W_arr, target_rank)
                
                if err < 1e-6:
                    # Try to round to integers and verify exactly
                    U_int = [[int(round(x)) for x in row] for row in U_arr.tolist()]
                    V_int = [[int(round(x)) for x in row] for row in V_arr.tolist()]
                    W_int = [[int(round(x)) for x in row] for row in W_arr.tolist()]
                    if check_decomposition_exact(U_int, V_int, W_int, target_rank):
                        best_rank = target_rank
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        found = True
                        break
                
                if err < best_err * 0.999:
                    best_err = err
                
                # Perturb a random element
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, target_rank - 1)
                    U_arr[row, col] += random.choice([-1, 1])
                elif mat_idx == 1:
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, target_rank - 1)
                    V_arr[row, col] += random.choice([-1, 1])
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, target_rank - 1)
                    W_arr[row, col] += random.choice([-1, 1])
            
            if found:
                break
        
        if found:
            break
    
    # Final check: if we found something better, return it; else return schoolbook
    if best_rank < rank_school:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
