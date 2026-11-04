"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
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
    
    # Precompute index arrays for fast verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Fast vectorized check of decomposition."""
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        return np.max(np.abs(total - 1.0)) <= tol
    
    def compute_residual(U, V, W, rank):
        """Compute residual for optimization."""
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        
        return total - 1.0
    
    # Value set for exactness
    value_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Strategy 1: Use commutativity - M_{3,3,8} is isomorphic to M_{8,3,3}
    # For M_{8,3,3}: multiply 8x3 by 3x3. Use block structure with Strassen-like decomposition.
    # Key insight: T_{3,3,8}[i,j,k] = T_{8,3,3}[k,j,i]
    # We can find decomposition for M_{8,3,3} and permute indices.
    
    # For M_{8,3,3}: think of 8x3 matrix as having 8 rows, 3 cols.
    # Split 8 into 2+2+2+2 or use other block decompositions.
    
    # Try structured decomposition exploiting that p=8 = 2^3
    # Use recursive Strassen-like approach
    
    def strassen_3x3_block(A_size, B_size, C_size, depth=0):
        """Generate Strassen-style decomposition for small matrices."""
        # For 3x3 x 3x3, use known rank-7 decomposition
        if A_size == 3 and B_size == 3 and C_size == 3:
            # Strassen decomposition for 3x3 (rank 7)
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # This gives 7 multiplications for 3x3x3
            return 7
        return A_size * B_size * C_size
    
    # Try direct search with better initialization
    # Use the fact that the problem has structure: for each (i,k), sum_j U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    for rank in range(21, 72):
        if time.time() > deadline - 10:
            break
        
        max_attempts = max(5, min(30, 150 - rank * 2))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structure: each rank dimension handles a subset of (i,j,k) triples
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            # Structured initialization: assign rank dimensions to blocks
            # For M_{3,3,8}, use block structure where each j-index has its own rank block
            rank_per_j = rank // m
            rem = rank % m
            
            for j in range(m):
                start_r = j * rank_per_j + min(j, rem)
                size_r = rank_per_j + (1 if j < rem else 0)
                
                for r in range(start_r, start_r + size_r):
                    # For this rank dimension, only certain (i,k) pairs are active
                    # Random but structured assignment
                    for i in range(n):
                        if random.random() < 0.5:
                            U[i * m + j, r] = random.choice([1, -1, 0.5, -0.5])
                    for k in range(p):
                        if random.random() < 0.5:
                            V[j * p + k, r] = random.choice([1, -1, 0.5, -0.5])
                    for i in range(n):
                        if random.random() < 0.5:
                            W[k * n + i, r] = random.choice([1, -1, 0.5, -0.5])
            
            # Check if valid
            if check_decomposition(U, V, W, rank, tol=1e-6):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_rank = rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
            
            # Local search polish with gradient-like updates
            current_residual = np.max(np.abs(compute_residual(U, V, W, rank)))
            
            if current_residual < 2.0:
                for polish_iter in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Pick the matrix and position with largest contribution to error
                    mat_choice = random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        mat = U
                        rows = n * m
                    elif mat_choice == 'V':
                        mat = V
                        rows = m * p
                    else:
                        mat = W
                        rows = p * n
                    
                    row = random.randint(0, rows - 1)
                    col = random.randint(0, rank - 1)
                    
                    old_val = mat[row, col]
                    best_val = old_val
                    best_res = current_residual
                    
                    for delta in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]:
                        mat[row, col] = old_val + delta
                        res = np.max(np.abs(compute_residual(U, V, W, rank)))
                        if res < best_res:
                            best_res = res
                            best_val = mat[row, col]
                    
                    mat[row, col] = best_val
                    current_residual = best_res
                    
                    if current_residual < 1e-8:
                        break
                
                if current_residual < 1e-6:
                    # Snap to nearest value
                    U_snapped = np.zeros_like(U)
                    V_snapped = np.zeros_like(V)
                    W_snapped = np.zeros_like(W)
                    for mat, snapped in [(U, U_snapped), (V, V_snapped), (W, W_snapped)]:
                        for i in range(len(mat)):
                            for j in range(len(mat[0])):
                                closest = min(value_set, key=lambda v: abs(v - mat[i, j]))
                                snapped[i, j] = closest
                    
                    if check_decomposition(U_snapped, V_snapped, W_snapped, rank, tol=1e-6):
                        U_list = [[float(x) for x in row] for row in U_snapped]
                        V_list = [[float(x) for x in row] for row in V_snapped]
                        W_list = [[float(x) for x in row] for row in W_snapped]
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
