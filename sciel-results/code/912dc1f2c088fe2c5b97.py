"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

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
    
    # Strategy: Use block structure with p=8 split into [0,1,2], [3,4,5], [6,7]
    # For each 3x3x3 block, try Strassen-like rank 7
    # For 3x3x2 block, use schoolbook rank 6
    # Target: 7 + 7 + 6 = 20 (optimistic), more realistically 9+9+6=24
    
    # Precompute Strassen formulas for 3x3x3
    # Standard Strassen for 2x2 gives rank 7. For 3x3, we can use block structure.
    # A 3x3 matrix multiplication can be decomposed using 3x3 blocks of 1x1,
    # but that's just schoolbook rank 9.
    # Alternative: use the known rank-7 formula for 3x3? Actually Strassen's
    # original was for 2x2. For 3x3, the minimal rank is unknown but bounded.
    
    # Let's try a different approach: use the fact that we can merge some terms
    # by exploiting linearity.
    
    # Approach: Start with schoolbook and try to merge pairs of terms
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged if
    # we can find u1,v1,w1,u2,v2,w2 such that the sum works.
    
    # More practical: use random local search with structured initialization
    
    best_found = None
    
    # Try target ranks from 68 down to 60
    for target_rank in range(68, 59, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 8 if target_rank <= 65 else 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with structured pattern
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Block structure:
            # Block 1: k in [0, 2] - use first ~target_rank/3 terms
            # Block 2: k in [3, 5] - use next ~target_rank/3 terms
            # Block 3: k in [6, 7] - use remaining terms (schoolbook)
            
            block1_size = min(9, max(7, target_rank // 3))
            block2_size = min(9, max(7, (target_rank - block1_size) // 2))
            block3_size = target_rank - block1_size - block2_size
            
            # Ensure block3 has at least 6 terms for schoolbook of 3x3x2
            if block3_size < 6:
                # Redistribute
                block1_size = max(7, (target_rank - 6) // 2)
                block2_size = max(7, target_rank - 6 - block1_size)
                block3_size = target_rank - block1_size - block2_size
            
            # Initialize Block 1 (k=0,1,2) with random +/-1 pattern
            for r in range(block1_size):
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.7:
                            U[i * m + j][r] = random.choice([-1, 1])
                for j in range(m):
                    for k in range(3):
                        if random.random() < 0.7:
                            V[j * p + k][r] = random.choice([-1, 1])
                for k in range(3):
                    for i in range(n):
                        if random.random() < 0.7:
                            W[k * n + i][r] = random.choice([-1, 1])
            
            # Initialize Block 2 (k=3,4,5)
            for r in range(block1_size, block1_size + block2_size):
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.7:
                            U[i * m + j][r] = random.choice([-1, 1])
                for j in range(m):
                    for k in range(3, 6):
                        if random.random() < 0.7:
                            V[j * p + k][r] = random.choice([-1, 1])
                for k in range(3, 6):
                    for i in range(n):
                        if random.random() < 0.7:
                            W[k * n + i][r] = random.choice([-1, 1])
            
            # Initialize Block 3 (k=6,7) with schoolbook pattern
            for r in range(block1_size + block2_size, target_rank):
                idx = r - (block1_size + block2_size)
                if idx < 6:
                    i = idx % 3
                    j = (idx // 3) % 3
                    k = 6 + (idx // 9)
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
            
            # Local search with gradient-like updates
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            best_err = compute_error_vectorized(U_arr, V_arr, W_arr, target_rank)
            
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                
                err = compute_error_vectorized(U_arr, V_arr, W_arr, target_rank)
                
                if err < 1e-4:
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
                
                # Perturb a random element in a block-appropriate way
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, target_rank - 1)
                    if col < block1_size:
                        U_arr[row, col] += random.choice([-1, 1])
                    elif col < block1_size + block2_size:
                        U_arr[row, col] += random.choice([-1, 1])
                    # Block 3: keep schoolbook, don't perturb
                elif mat_idx == 1:
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, target_rank - 1)
                    k_local = row % p
                    if col < block1_size:
                        if k_local < 3:
                            V_arr[row, col] += random.choice([-1, 1])
                    elif col < block1_size + block2_size:
                        if 3 <= k_local < 6:
                            V_arr[row, col] += random.choice([-1, 1])
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, target_rank - 1)
                    k_local = row // n
                    if col < block1_size:
                        if k_local < 3:
                            W_arr[row, col] += random.choice([-1, 1])
                    elif col < block1_size + block2_size:
                        if 3 <= k_local < 6:
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
