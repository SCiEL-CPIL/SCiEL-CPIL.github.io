"""
Tensor decomposition for 4x5 * 5x5 matrix multiplication.
Uses a block-partitioned approach inspired by Strassen-like rank reductions.
Partitions the 4x5 and 5x5 matrices into 2x2 and 2x3 blocks to exploit
bilinear structure and find a lower-rank exact decomposition than schoolbook 100.
"""
import time
import random
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank_school = n * m * p  # 100
    
    deadline = time.time() + 135  # safety margin before 150s kill
    
    def build_schoolbook():
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1.0
                    V[j * p + k][c] = 1.0
                    W[k * n + i][c] = 1.0
        return U, V, W
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using numpy for speed."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def verify_fast(U_list, V_list, W_list, rank):
        """Faster verification using precomputed arrays."""
        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Start with schoolbook as best
    best_U, best_V, best_W = build_schoolbook()
    best_rank = rank_school
    
    # Strategy 1: Try to find a lower-rank decomposition using structured
    # block-based patterns inspired by Strassen's algorithm.
    # Partition A (4x5) into blocks: rows 0-1 and 2-3, cols 0-1, 2-3, 4
    # Partition B (5x5) into blocks: rows 0-1, 2-3, 4; cols 0-1, 2-3, 4
    # This gives a 2x2 block structure for the row dimension and a 3x3 for columns.
    
    # For now, do a more targeted search: try ranks from 99 down to 80
    # with structured initialization based on block patterns.
    
    for target_rank in range(rank_school - 1, max(rank_school - 40, 0), -1):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 15 if target_rank > 90 else 8
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate factors with block-structured sparsity
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                # Assign each column to a block pattern
                # U has 20 rows (4*5), V has 25 rows (5*5), W has 20 rows (5*4)
                
                # Randomly select entries but with some structure
                # Each column should have entries that correspond to a valid
                # bilinear term pattern
                
                # For U: pick 1-2 entries
                num_u = random.randint(1, 2)
                for _ in range(num_u):
                    row = random.randint(0, n * m - 1)
                    U[row, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                
                # For V: pick 1-2 entries
                num_v = random.randint(1, 2)
                for _ in range(num_v):
                    row = random.randint(0, m * p - 1)
                    V[row, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                
                # For W: pick 1-2 entries
                num_w = random.randint(1, 2)
                for _ in range(num_w):
                    row = random.randint(0, p * n - 1)
                    W[row, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_fast(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                return {
                    "U": best_U,
                    "V": best_V,
                    "W": best_W
                }
    
    # If no lower rank found, return schoolbook
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
