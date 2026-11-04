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
    
    # Precompute index arrays for fast evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition_fast(U, V, W, rank):
        """Fast check using vectorized operations."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        if np.max(np.abs(total - 1.0)) > 1e-6:
            return False
        return True
    
    def check_decomposition_exact(U, V, W, rank):
        """Exact check using integer arithmetic."""
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        total = np.zeros(n * m * p, dtype=np.int64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        return np.all(total == 1)
    
    # Strategy: Use Strassen-like decomposition for 3x3 x 3x8
    # For 3x3 x 3x3, best known rank is 23.
    # For 3x3 x 3x8, we can use a block approach:
    # Split the 8 columns into blocks and use efficient decompositions.
    
    # Approach: Use a structured decomposition based on polynomial method or
    # known low-rank constructions.
    
    # First, try a deterministic construction based on Strassen's algorithm
    # extended to rectangular matrices.
    
    # For 3x3 x 3x3, use the rank-23 decomposition.
    # For 3x3 x 3x8, we can use 2 blocks of 3x3 and 1 block of 3x2,
    # or find a unified decomposition.
    
    # Let's try a systematic search with structured initialization.
    # Use values from a larger set: {-2, -1, -0.5, 0, 0.5, 1, 2}
    
    value_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    value_set_int = [-2, -1, 0, 1, 2]
    
    # Try ranks from 36 upward with better search
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
                # Verify with exact arithmetic
                U_int = [[int(round(x * 2)) / 2 for x in row] for row in U]
                V_int = [[int(round(x * 2)) / 2 for x in row] for row in V]
                W_int = [[int(round(x * 2)) / 2 for x in row] for row in W]
                
                # Check if all values are half-integers
                valid = True
                for mat in [U_int, V_int, W_int]:
                    for row in mat:
                        for val in row:
                            if abs(val * 2 - round(val * 2)) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid and check_decomposition_fast(U_int, V_int, W_int, rank):
                    best_rank = rank
                    best_result = {"U": U_int, "V": V_int, "W": W_int}
                    return best_result
    
    # Approach: Use a local search starting from schoolbook.
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
    
    # Approach: Try a structured decomposition based on Kronecker products
    # For 3x3 x 3x8, we can use the fact that 8 = 3 + 3 + 2
    # and combine decompositions for each block.
    
    if time.time() < deadline - 10:
        # Try to construct a decomposition using block structure
        # Split the 8 columns into 3+3+2 blocks
        
        # For 3x3 x 3x3, use a rank-23 decomposition (if we can find one)
        # For 3x3 x 3x2, use a rank-18 decomposition (schoolbook)
        # Total rank would be 23 + 23 + 18 = 64, which is better than 72
        
        # But we need to find the actual decompositions.
        # Let's try a more direct approach.
        
        # Use a polynomial method: represent the matrix multiplication
        # as evaluating a polynomial at specific points.
        
        # For 3x3 x 3x8, we can use 8 evaluation points.
        # The rank would be related to the degree of the polynomial.
        
        # This is complex, so let's try a simpler approach.
        
        # Try to find a decomposition where U, V, W have a specific structure
        # based on the indices.
        
        # For each (i,j,k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # One approach: use a basis of functions that are separable in i, j, k.
        
        # Let's try a structured random search with more attempts.
        for rank in range(36, 72):
            if time.time() > deadline - 5:
                break
            
            # Try more attempts for lower ranks
            max_attempts = max(100, 300 - (rank - 36) * 3)
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Use a structured initialization based on index patterns
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                # Use a mixture of structured and random patterns
                for r in range(rank):
                    # Each rank component has a specific structure
                    # based on the rank index
                    struct_type = r % 3
                    
                    if struct_type == 0:
                        # Pattern based on i
                        for i in range(n):
                            for j in range(m):
                                U[i * m + j][r] = random.choice([-1, 0, 1])
                        for j in range(m):
                            for k in range(p):
                                V[j * p + k][r] = random.choice([-1, 0, 1])
                        for k in range(p):
                            for i in range(n):
                                W[k * n + i][r] = random.choice([-1, 0, 1])
                    elif struct_type == 1:
                        # Pattern based on j
                        for i in range(n):
                            for j in range(m):
                                U[i * m + j][r] = random.choice([-1, 0, 1])
                        for j in range(m):
                            for k in range(p):
                                V[j * p + k][r] = random.choice([-1, 0, 1])
                        for k in range(p):
                            for i in range(n):
                                W[k * n + i][r] = random.choice([-1, 0, 1])
                    else:
                        # Pattern based on k
                        for i in range(n):
                            for j in range(m):
                                U[i * m + j][r] = random.choice([-1, 0, 1])
                        for j in range(m):
                            for k in range(p):
                                V[j * p + k][r] = random.choice([-1, 0, 1])
                        for k in range(p):
                            for i in range(n):
                                W[k * n + i][r] = random.choice([-1, 0, 1])
                
                if check_decomposition_fast(U, V, W, rank):
                    # Verify with exact arithmetic
                    U_int = [[int(round(x)) for x in row] for row in U]
                    V_int = [[int(round(x)) for x in row] for row in V]
                    W_int = [[int(round(x)) for x in row] for row in W]
                    
                    if check_decomposition_fast(U_int, V_int, W_int, rank):
                        best_rank = rank
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
