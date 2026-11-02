"""Tensor decomposition search for 3x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A (3x5), B (5x5), C (3x5)
    # Schoolbook rank is n*m*p = 75.
    # We attempt to find a lower rank decomposition.
    # Known lower bounds suggest rank 11 or 12 is possible.
    # We will use a randomized local search to refine a random initialization
    # towards an exact integer/half-integer solution.
    
    deadline = time.time() + 140.0  # Safety margin
    
    # Helper to check if a decomposition is valid
    def check_valid(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] == 1 for all valid i,j,k
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = np.array(U).reshape(n, m, rank)
        V_reshaped = np.array(V).reshape(m, p, rank)
        W_reshaped = np.array(W).reshape(p, n, rank)
        
        # Compute tensor T
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # Use einsum for efficiency
        # U[i,j,c] * V[j,k,c] -> (i, j, k, c) ? No, sum over c.
        # Let's do it step by step or use einsum.
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # We want T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # This is a bit tricky with standard einsum because indices are permuted.
        # Let's just loop over i, j, k since n,m,p are small (3,5,5).
        # Total 75 checks.
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Helper to snap to nearest half-integer
    def snap_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    # 1. Initialize with Schoolbook Decomposition (Guaranteed Valid)
    schoolbook_rank = n * m * p
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    best_rank = schoolbook_rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # 2. Search for Lower Rank Decompositions
    # We try ranks from 12 down to 10.
    # For each rank, we do random restarts.
    
    def try_find_decomposition(rank, max_iterations=2000):
        """
        Attempts to find a rank-r decomposition using random initialization and local search.
        Returns (U, V, W) if found, else None.
        """
        for _ in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Random initialization with small integers/half-integers
            # Values in {-1, -0.5, 0, 0.5, 1, 1.5, 2}
            U_rand = np.random.randint(-4, 5, (n * m, rank)) / 2.0
            V_rand = np.random.randint(-4, 5, (m * p, rank)) / 2.0
            W_rand = np.random.randint(-4, 5, (p * n, rank)) / 2.0
            
            # Check if this random guess happens to be a solution (unlikely but possible)
            if check_valid(U_rand, V_rand, W_rand, rank):
                return U_rand, V_rand, W_rand
            
            # Local search: perturb one entry at a time and see if error decreases
            # We need a metric for error.
            # Error = sum_{i,j,k} (T[i,j,k] - 1)^2
            
            def compute_error(U, V, W, rank):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (val - 1.0) ** 2
                return err

            U_curr = U_rand.copy()
            V_curr = V_rand.copy()
            W_curr = W_rand.copy()
            
            current_error = compute_error(U_curr, V_curr, W_curr, rank)
            
            # If initial error is already 0, we found it
            if current_error < 1e-6:
                return U_curr, V_curr, W_curr
            
            # Local search steps
            steps_per_iter = 10
            for _ in range(steps_per_iter):
                if time.time() > deadline:
                    return None
                    
                # Pick a random matrix and a random entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    row, col_idx = np.random.randint(n * m), np.random.randint(rank)
                    # Try changing U_curr[row, col_idx]
                    old_val = U_curr[row, col_idx]
                    # Try nearby half-integers
                    candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0]
                    best_candidate = None
                    best_candidate_err = current_error
                    
                    for cand in candidates:
                        U_curr[row, col_idx] = cand
                        new_err = compute_error(U_curr, V_curr, W_curr, rank)
                        if new_err < best_candidate_err:
                            best_candidate_err = new_err
                            best_candidate = cand
                    U_curr[row, col_idx] = best_candidate if best_candidate is not None else old_val
                    
                elif mat_idx == 1:
                    row, col_idx = np.random.randint(m * p), np.random.randint(rank)
                    old_val = V_curr[row, col_idx]
                    candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0]
                    best_candidate = None
                    best_candidate_err = current_error
                    for cand in candidates:
                        V_curr[row, col_idx] = cand
                        new_err = compute_error(U_curr, V_curr, W_curr, rank)
                        if new_err < best_candidate_err:
                            best_candidate_err = new_err
                            best_candidate = cand
                    V_curr[row, col_idx] = best_candidate if best_candidate is not None else old_val
                else:
                    row, col_idx = np.random.randint(p * n), np.random.randint(rank)
                    old_val = W_curr[row, col_idx]
                    candidates = [old_val + 0.5, old_val - 0.5, old_val + 1.0, old_val - 1.0]
                    best_candidate = None
                    best_candidate_err = current_error
                    for cand in candidates:
                        W_curr[row, col_idx] = cand
                        new_err = compute_error(U_curr, V_curr, W_curr, rank)
                        if new_err < best_candidate_err:
                            best_candidate_err = new_err
                            best_candidate = cand
                    W_curr[row, col_idx] = best_candidate if best_candidate is not None else old_val
                
                # Update current error
                # Optimization: Only recompute full error if a change was made
                # For simplicity, recompute. n*m*p*rank is small.
                current_error = compute_error(U_curr, V_curr, W_curr, rank)
                
                if current_error < 1e-6:
                    # Snap to half-integers to ensure exactness
                    U_curr = snap_to_half_int(U_curr)
                    V_curr = snap_to_half_int(V_curr)
                    W_curr = snap_to_half_int(W_curr)
                    if check_valid(U_curr, V_curr, W_curr, rank):
                        return U_curr, V_curr, W_curr
                    else:
                        # If snapping broke it, keep the float version? 
                        # The problem requires integer/half-integer entries.
                        # So we must return snapped version.
                        # If snapped version is invalid, this path failed.
                        pass

        return None

    # Try ranks 12, 11, 10
    for rank in [12, 11, 10]:
        if time.time() > deadline:
            break
            
        result = try_find_decomposition(rank, max_iterations=50)
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            break

    # If no better rank found, return schoolbook
    if best_rank == schoolbook_rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
