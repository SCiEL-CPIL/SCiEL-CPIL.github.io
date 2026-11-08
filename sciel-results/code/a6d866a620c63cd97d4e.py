"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix
    by a 4x7 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 112).
    2. Attempt to reduce the rank using vectorized numpy operations for faster validation.
    3. Use targeted local search with smarter move selection.
    4. Add a polish pass for near-solutions.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Pre-compute index arrays for vectorized evaluation
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_indices = len(indices)
    
    # For each index, store the row indices into U, V, W
    u_rows = np.array([i * m + j for i, j, k in indices])
    v_rows = np.array([j * p + k for i, j, k in indices])
    w_rows = np.array([k * n + i for i, j, k in indices])
    
    # Schoolbook fallback (rank = n*m*p = 112)
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
    
    best_solution = None
    best_rank = rank_school
    
    def compute_errors_np(U_np, V_np, W_np, R):
        """Compute all tensor values using numpy broadcasting."""
        # U_np: (n*m, R), V_np: (m*p, R), W_np: (p*n, R)
        # For each (i,j,k), compute sum_r U[u_rows[r], r] * V[v_rows[r], r] * W[w_rows[r], r]
        U_sel = U_np[u_rows]  # (num_indices, R)
        V_sel = V_np[v_rows]  # (num_indices, R)
        W_sel = W_np[w_rows]  # (num_indices, R)
        products = U_sel * V_sel * W_sel  # (num_indices, R)
        sums = products.sum(axis=1)  # (num_indices,)
        errors = sums - 1  # Should be 0 for valid
        return errors
    
    def count_nonzero(errors, tol=1e-9):
        """Count how many entries are not close to zero."""
        return np.count_nonzero(np.abs(errors) > tol)
    
    # Try ranks from lower bound up to schoolbook rank
    # Lower bound is max(n*m, m*p, n*p) = 28
    ranks_to_try = [28, 30, 32, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        if R >= best_rank:
            continue
            
        # Try multiple restarts for this rank
        max_restarts = 30 if R < 40 else 10 if R < 60 else 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with structured random values
            # Use a mix of 0, 1, -1 with some structure
            U_np = np.zeros((n * m, R), dtype=np.int64)
            V_np = np.zeros((m * p, R), dtype=np.int64)
            W_np = np.zeros((p * n, R), dtype=np.int64)
            
            # Initialize with random small integers
            for r in range(R):
                # Each term should contribute to some subset of indices
                # Randomly assign non-zero entries
                for _ in range(random.randint(1, 3)):
                    U_np[random.randint(0, n * m - 1), r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    V_np[random.randint(0, m * p - 1), r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    W_np[random.randint(0, p * n - 1), r] = random.choice([-1, 1])
            
            # Local search
            max_iters = 2000
            prev_error_count = None
            stuck_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                errors = compute_errors_np(U_np, V_np, W_np, R)
                error_count = count_nonzero(errors)
                
                if error_count == 0:
                    # Valid solution found!
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_np.tolist()
                    best_rank = R
                    best_solution = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
                
                # Track progress
                if prev_error_count is not None:
                    if error_count < prev_error_count:
                        stuck_count = 0
                    else:
                        stuck_count += 1
                prev_error_count = error_count
                
                if stuck_count > 50:
                    # Stuck, try a larger perturbation
                    stuck_count = 0
                    # Randomly reset a few entries
                    for _ in range(random.randint(1, 3)):
                        choice = random.randint(0, 2)
                        if choice == 0:
                            idx = random.randint(0, n * m - 1)
                            col = random.randint(0, R - 1)
                            U_np[idx, col] = random.choice([-2, -1, 0, 1, 2])
                        elif choice == 1:
                            idx = random.randint(0, m * p - 1)
                            col = random.randint(0, R - 1)
                            V_np[idx, col] = random.choice([-2, -1, 0, 1, 2])
                        else:
                            idx = random.randint(0, p * n - 1)
                            col = random.randint(0, R - 1)
                            W_np[idx, col] = random.choice([-2, -1, 0, 1, 2])
                    continue
                
                # Find the worst error
                error_indices = np.where(np.abs(errors) > 1e-9)[0]
                if len(error_indices) == 0:
                    continue
                    
                # Pick a random error index
                err_idx = random.choice(error_indices)
                i, j, k = indices[err_idx]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                current_sum = int(errors[err_idx] + 1)  # errors = sums - 1, so sums = errors + 1
                
                # Try to fix by adjusting one variable in a random term r
                r = random.randint(0, R - 1)
                
                u_r = int(U_np[u_idx, r])
                v_r = int(V_np[v_idx, r])
                w_r = int(W_np[w_idx, r])
                contrib = u_r * v_r * w_r
                
                # We want new sum = 1
                # new_sum = current_sum - contrib + new_contrib
                # 1 = current_sum - contrib + new_contrib
                # new_contrib = 1 - current_sum + contrib
                target_contrib = 1 - current_sum + contrib
                
                # Try U
                denom = v_r * w_r
                if denom != 0:
                    if target_contrib % denom == 0:
                        new_u = target_contrib // denom
                        if -10 <= new_u <= 10 and new_u != u_r:
                            U_np[u_idx, r] = new_u
                            continue
                
                # Try V
                denom = u_r * w_r
                if denom != 0:
                    if target_contrib % denom == 0:
                        new_v = target_contrib // denom
                        if -10 <= new_v <= 10 and new_v != v_r:
                            V_np[v_idx, r] = new_v
                            continue
                
                # Try W
                denom = u_r * v_r
                if denom != 0:
                    if target_contrib % denom == 0:
                        new_w = target_contrib // denom
                        if -10 <= new_w <= 10 and new_w != w_r:
                            W_np[w_idx, r] = new_w
                            continue
                
                # If no simple fix was found, try a random perturbation
                choice = random.randint(0, 2)
                if choice == 0:
                    idx = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    U_np[idx, col] += random.choice([-1, 1])
                elif choice == 1:
                    idx = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    V_np[idx, col] += random.choice([-1, 1])
                else:
                    idx = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    W_np[idx, col] += random.choice([-1, 1])
            
            if found:
                break
        
        if found:
            break
    
    # Polish pass: if we found a solution, try to verify and potentially improve
    if best_solution:
        # Verify the solution
        U_np = np.array(best_solution['U'])
        V_np = np.array(best_solution['V'])
        W_np = np.array(best_solution['W'])
        R = len(U_np[0])
        errors = compute_errors_np(U_np, V_np, W_np, R)
        if count_nonzero(errors) == 0:
            return best_solution
    
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
