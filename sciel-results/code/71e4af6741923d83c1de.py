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
    deadline = start_time + 140  # safety margin before 150s limit

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

    # Strategy: Use Strassen-Winograd recursive decomposition
    # Key insight: 3x3x8 can be decomposed using block structure
    # View the 3x8 matrix as having 8 columns, split into blocks
    # Use the fact that matrix multiplication is bilinear and can be decomposed
    
    # For 3x3 x 3x8, we can view this as:
    # C = A * B where A is 3x3, B is 3x8
    # Split B into B1 (3x4) and B2 (3x4)
    # C = A*B1 + A*B2
    # But this doesn't directly help with rank reduction
    
    # Better approach: Use the Strassen algorithm for 2x2x2 blocks
    # The 3x3x8 problem can be decomposed into smaller subproblems
    
    # Let's try a direct approach: exploit the structure that for each fixed j,
    # we have a 3x1x8 subproblem (matrix-vector multiplication)
    
    # For matrix-vector multiplication (3x3 x 3x1), the rank is known to be 9
    # (same as schoolbook) for this specific case
    
    # However, we can try to find a lower rank decomposition by:
    # 1. Using random search with structured initialization
    # 2. Using local search to refine
    
    # Try to construct a decomposition using block structure
    # Split the 8 columns into groups and use Strassen-like methods
    
    # For now, let's try a more systematic search approach
    # with better initialization based on known patterns
    
    # Try ranks from 56 to 71 (slightly below schoolbook)
    for target_rank in range(68, 72):
        if time.time() > deadline - 15:
            break
        
        max_attempts = max(10, min(50, 180 - target_rank * 2))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structured pattern
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Structured initialization: assign rank dimensions to handle subsets
            # Use a pattern where each rank dimension handles a specific (i,j,k) combination
            # with some sharing to reduce rank
            
            # Pattern 1: Block-based assignment
            for r in range(target_rank):
                # Determine which (i,j,k) this rank dimension handles
                # Use a systematic pattern based on r
                i = (r // (m * p)) % n
                j = (r // p) % m
                k = r % p
                
                # Set values with some randomness to allow for cancellation
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                
                # Add some perturbation to allow for rank reduction
                if r < target_rank - 4:
                    # Add small perturbations to enable cancellation
                    if random.random() < 0.3:
                        U[i * m + j, r] = random.choice([0.5, -0.5])
                    if random.random() < 0.3:
                        V[j * p + k, r] = random.choice([0.5, -0.5])
                    if random.random() < 0.3:
                        W[k * n + i, r] = random.choice([0.5, -0.5])
            
            # Check if valid
            if check_decomposition(U, V, W, target_rank, tol=1e-6):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                return best_result
            
            # Local search polish
            current_residual = np.max(np.abs(compute_residual(U, V, W, target_rank)))
            
            if current_residual < 3.0:
                for polish_iter in range(100):
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
                    col = random.randint(0, target_rank - 1)
                    
                    old_val = mat[row, col]
                    best_val = old_val
                    best_res = current_residual
                    
                    for delta in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]:
                        mat[row, col] = old_val + delta
                        res = np.max(np.abs(compute_residual(U, V, W, target_rank)))
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
                    
                    if check_decomposition(U_snapped, V_snapped, W_snapped, target_rank, tol=1e-6):
                        U_list = [[float(x) for x in row] for row in U_snapped]
                        V_list = [[float(x) for x in row] for row in V_snapped]
                        W_list = [[float(x) for x in row] for row in W_snapped]
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        return best_result
    
    # Try a more sophisticated approach: use the fact that we can share
    # rank dimensions across different (i,j,k) triples if the values allow cancellation
    
    # For each pair of (i1,j1,k1) and (i2,j2,k2), we can potentially share
    # a rank dimension if we can set values such that the contributions cancel
    # for unwanted triples and add for wanted triples
    
    # This is a complex constraint satisfaction problem, but we can try
    # a greedy approach: start with schoolbook and try to merge pairs
    
    if time.time() < deadline - 20:
        # Try to merge rank dimensions
        U_merge = np.array(U_school, dtype=np.float64)
        V_merge = np.array(V_school, dtype=np.float64)
        W_merge = np.array(W_school, dtype=np.float64)
        
        # Try to merge pairs of columns
        merged = False
        for r1 in range(rank_school):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, rank_school):
                if time.time() > deadline - 5:
                    break
                
                # Try to merge r1 and r2 by setting values to allow cancellation
                # This requires solving a system of equations, which is complex
                # For now, just try a few random merges
                
                if random.random() < 0.1:
                    # Attempt merge: set values so that for most triples,
                    # the contributions cancel
                    U_new = U_merge.copy()
                    V_new = V_merge.copy()
                    W_new = W_merge.copy()
                    
                    # Try to merge by setting one column to be a linear combination
                    # of the other
                    if random.random() < 0.5:
                        U_new[:, r2] = U_new[:, r1] * random.choice([0.5, -0.5, 1.0, -1.0])
                        V_new[:, r2] = V_new[:, r1] * random.choice([0.5, -0.5, 1.0, -1.0])
                        W_new[:, r2] = W_new[:, r1] * random.choice([0.5, -0.5, 1.0, -1.0])
                    else:
                        U_new[:, r2] = -U_new[:, r1]
                        V_new[:, r2] = V_new[:, r1]
                        W_new[:, r2] = W_new[:, r1]
                    
                    # Check if the merge is valid
                    if check_decomposition(U_new, V_new, W_new, rank_school, tol=1e-6):
                        # Remove the merged column
                        U_new = np.delete(U_new, r2, axis=1)
                        V_new = np.delete(V_new, r2, axis=1)
                        W_new = np.delete(W_new, r2, axis=1)
                        
                        if check_decomposition(U_new, V_new, W_new, rank_school - 1, tol=1e-6):
                            U_merge = U_new
                            V_merge = V_new
                            W_merge = W_new
                            merged = True
        
        if merged:
            final_rank = U_merge.shape[1]
            if final_rank < best_rank:
                U_list = [[float(x) for x in row] for row in U_merge]
                V_list = [[float(x) for x in row] for row in V_merge]
                W_list = [[float(x) for x in row] for row in W_merge]
                best_rank = final_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
