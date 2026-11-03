import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
R_SCHOOLBOOK = n * m * p  # 36
START_RANK = 23  # Known best upper bound for 3x3x4 matrix multiplication

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Initialize with schoolbook as fallback
    best_rank = R_SCHOOLBOOK
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    # Precompute index mappings for fast error computation
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] == 1 for all i,j,k
    
    # For fast incremental error update:
    # If we change U[row, col], it affects T[i,j,k] where i*m+j == row
    # i = row // m, j = row % m, and k ranges over [0, p)
    
    def compute_full_error(U, V, W, R):
        """Compute total squared error using vectorized operations."""
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum for efficient computation
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        err = np.sum((T - 1.0) ** 2)
        return err

    def compute_error_for_rank(R, U, V, W):
        """Compute error for given rank and matrices."""
        return compute_full_error(U, V, W, R)

    def local_search(U, V, W, R, time_limit):
        """
        Perform local search to reduce error to near-zero.
        Returns (success, U, V, W, final_error)
        """
        start_time = time.time()
        current_err = compute_full_error(U, V, W, R)
        
        if current_err < 1e-9:
            return True, U, V, W, current_err

        num_entries_U = n * m
        num_entries_V = m * p
        num_entries_W = p * n
        total_entries = num_entries_U + num_entries_V + num_entries_W

        # Precompute for fast partial error updates
        # We'll maintain the current T tensor
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

        max_iterations = 50000
        for it in range(max_iterations):
            if time.time() > time_limit:
                break
            if current_err < 1e-9:
                break

            # Pick a random matrix and entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row = np.random.randint(num_entries_U)
                col = np.random.randint(R)
                i = row // m
                j = row % m
                old_val = U[row, col]
            elif mat_idx == 1:
                row = np.random.randint(num_entries_V)
                col = np.random.randint(R)
                j = row // p
                k = row % p
                old_val = V[row, col]
            else:
                row = np.random.randint(num_entries_W)
                col = np.random.randint(R)
                k = row // n
                i = row % n
                old_val = W[row, col]

            # Try all 5 possible values
            values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            best_new_val = old_val
            best_err = current_err

            for new_val in values:
                if new_val == old_val:
                    continue

                # Apply change
                if mat_idx == 0:
                    U[row, col] = new_val
                    U_t[i, j, col] = new_val
                elif mat_idx == 1:
                    V[row, col] = new_val
                    V_t[j, k, col] = new_val
                else:
                    W[row, col] = new_val
                    W_t[k, i, col] = new_val

                # Recompute T efficiently - only affected terms
                # For mat_idx == 0: T[i,j,:] changes for all k
                # For mat_idx == 1: T[:,j,k] changes for all i
                # For mat_idx == 2: T[:, :, k] and specifically T[:,i,k] changes
                # Actually, let's just recompute T for efficiency since R is small
                
                T_new = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                new_err = np.sum((T_new - 1.0) ** 2)

                if new_err < best_err:
                    best_err = new_err
                    best_new_val = new_val
                    T = T_new
                else:
                    # Revert
                    if mat_idx == 0:
                        U[row, col] = old_val
                        U_t[i, j, col] = old_val
                    elif mat_idx == 1:
                        V[row, col] = old_val
                        V_t[j, k, col] = old_val
                    else:
                        W[row, col] = old_val
                        W_t[k, i, col] = old_val

            if best_new_val != old_val:
                current_err = best_err
                # The value is already applied if it was the best
                # But we need to make sure U_t, V_t, W_t are consistent
                # They should be since we updated them when applying
            else:
                # Ensure old value is restored in the reshaped views
                if mat_idx == 0:
                    U_t[i, j, col] = old_val
                elif mat_idx == 1:
                    V_t[j, k, col] = old_val
                else:
                    W_t[k, i, col] = old_val

        return current_err < 1e-9, U, V, W, current_err

    # Try ranks from START_RANK upward
    for target_rank in range(START_RANK, R_SCHOOLBOOK):
        if time.time() > deadline - 5:
            break

        # Time budget for this rank
        rank_time_budget = min(10, max(5, (deadline - time.time()) - 5))
        rank_deadline = time.time() + rank_time_budget

        found = False

        # Multiple restarts with different initializations
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > rank_deadline:
                break

            # Random initialization with half-integers
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, target_rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, target_rank))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, target_rank))

            # Quick check initial error
            init_err = compute_full_error(U, V, W, target_rank)
            if init_err < 1e-9:
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break

            # Run local search with time limit
            step_deadline = min(rank_deadline, time.time() + 2.0)
            success, U, V, W, final_err = local_search(U, V, W, target_rank, step_deadline)

            if success or final_err < 1e-9:
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break

            # If error is low but not zero, try a bit more
            if final_err < 0.01 and time.time() < rank_deadline - 1:
                success, U, V, W, final_err = local_search(U, V, W, target_rank, rank_deadline)
                if success or final_err < 1e-9:
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break

        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
