"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s out of 150s limit
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use block decomposition with known small matrix multiplication results
    # For 3x5x6, split the middle dimension 5 into 3+2
    
    # Approach 1: Direct systematic search with structured patterns
    # Try to find decompositions with rank < 90
    
    # Known: 3x3x3 can be done in rank 7 (Strassen)
    # 3x2x3 and 2x3x3 have rank 6 (schoolbook is optimal)
    
    # Let's try a block approach:
    # Split j (middle index) into j1 in [0,3) and j2 in [3,5)
    # C[i,k] = sum_{j1=0}^{2} A[i,j1]*B[j1,k] + sum_{j2=3}^{4} A[i,j2]*B[j2,k]
    
    # First part: 3x3x3 multiplication (A is 3x3, B is 3x6) - this is 3x3 by 3x6
    # Actually, this is 3x3 * 3x6 = 3x6, which is 3x3x6
    # 3x3x6: can be done as 2 * (3x3x3) if we split k, but that gives 2*7=14
    # Or we can do it directly. Rank of 3x3x6 >= max(9, 18, 9) = 18
    # Schoolbook for 3x3x6 is 54, but we can do better.
    
    # Actually, let me just try a smarter random search with more structure.
    
    # Generate candidate decompositions using known patterns
    # Pattern 1: All entries in {-1, 0, 1}
    # Pattern 2: All entries in {-1, -0.5, 0, 0.5, 1}
    
    import random
    random.seed(42)
    
    # Try ranks from 30 to 89
    # For each rank, try several random initializations with small integer entries
    
    # To speed up, use vectorized check
    def fast_check(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Vectorize over c
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j, :]  # shape (rank,)
                    v_val = V_arr[j * p + k, :]  # shape (rank,)
                    w_val = W_arr[k * n + i, :]  # shape (rank,)
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try various ranks
    # Lower bound: max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    # But actually, for matrix multiplication, a better lower bound is:
    # rank >= max(n*m, m*p, p*n) = 30
    # And also rank >= (n*m*p) / max(n,m,p) = 90/6 = 15 (weak)
    
    # Let's try ranks from 30 up, but be strategic
    # For small ranks, try more attempts; for larger ranks, fewer attempts
    
    ranks_to_try = list(range(30, 90))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            continue
        
        # Number of attempts based on rank
        if rank < 40:
            max_attempts = 100
        elif rank < 50:
            max_attempts = 50
        elif rank < 70:
            max_attempts = 20
        else:
            max_attempts = 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V, W with small integer/half-integer entries
            U_rand = np.zeros((n * m, rank), dtype=float)
            V_rand = np.zeros((m * p, rank), dtype=float)
            W_rand = np.zeros((p * n, rank), dtype=float)
            
            # Use sparse random initialization
            density = 0.3 if rank < 50 else 0.2
            
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < density:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        U_rand[i, c] = val
            
            for j in range(m * p):
                for c in range(rank):
                    if random.random() < density:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        V_rand[j, c] = val
            
            for k in range(p * n):
                for c in range(rank):
                    if random.random() < density:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        W_rand[k, c] = val
            
            # Check if this decomposes correctly
            if fast_check(U_rand, V_rand, W_rand, rank):
                # Convert to lists
                U_list = [[float(U_rand[i, c]) for c in range(rank)] for i in range(n * m)]
                V_list = [[float(V_rand[j, c]) for c in range(rank)] for j in range(m * p)]
                W_list = [[float(W_rand[k, c]) for c in range(rank)] for k in range(p * n)]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                print(f"Found rank {rank} decomposition!")
                break
    
    return best_result
# EVOLVE-BLOCK-END
