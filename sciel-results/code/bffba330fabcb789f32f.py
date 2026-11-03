"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit

    def calculate_cost(perm):
        # perm is a numpy array of locations
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    # Precompute
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    # Ensure we work with int64 for cost accumulation if possible, but float for Hungarian
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # Helper: efficient delta for swap (i, j)
    def calc_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        
        # Diagonal terms
        delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
        
        # Off-diagonal terms
        # Sum over k != i, j of (a[i,k] - a[j,k]) * (b[v,k] - b[u,k])
        # We can compute this as dot product of differences, then subtract i and j terms
        diff_A = a[i, :] - a[j, :]
        diff_B = b[v, perm] - b[u, perm]
        
        total_dot = np.dot(diff_A, diff_B)
        
        # Exclude i and j from the sum
        # Term for k=i: (a[i,i] - a[j,i]) * (b[v,i] - b[u,i])
        # Term for k=j: (a[i,j] - a[j,j]) * (b[v,j] - b[u,j])
        
        # Note: b[v, perm] is a vector where index k is b[v, perm[k]]
        # So diff_B[i] is b[v, perm[i]] - b[u, perm[i]] = b[v, u] - b[u, u]
        # And diff_A[i] is a[i, i] - a[j, i]
        
        exclude_i = diff_A[i] * diff_B[i]
        exclude_j = diff_A[j] * diff_B[j]
        
        off_diag_sum = total_dot - exclude_i - exclude_j
        delta += 2 * off_diag_sum
        
        return int(delta)

    # Tabu Search on a given starting permutation
    def tabu_search(start_p, max_time):
        current_p = start_p.copy()
        current_cost = calculate_cost(current_p)
        best_p = current_p.copy()
        best_cost = current_cost

        tabu_list = {}
        tabu_tenure = 80
        iteration = 0
        no_improve_count = 0
        max_no_improve = 200  # Lower threshold to trigger diversification sooner

        while time.time() - start_time < max_time:
            best_delta = 0
            best_i = -1
            best_j = -1

            # Evaluate all pairs, find best improving or best non-tabu move
            # To speed up, we can iterate in a random order or just standard
            # Standard is fine for n=256 (32k pairs)
            for i in range(n):
                if time.time() - start_time > max_time:
                    break
                for j in range(i + 1, n):
                    delta = calc_delta(i, j, current_p)
                    key1 = (i, j)
                    key2 = (j, i)
                    is_tabu = (key1 in tabu_list and tabu_list[key1] > iteration) or \
                              (key2 in tabu_list and tabu_list[key2] > iteration)

                    # Aspiration: accept tabu move if it improves best
                    if delta < best_delta:
                        if not is_tabu or (current_cost + delta < best_cost):
                            best_delta = delta
                            best_i = i
                            best_j = j

            if best_i >= 0:
                # Apply best move
                current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
                current_cost += best_delta
                tabu_list[(best_i, best_j)] = iteration + tabu_tenure
                tabu_list[(best_j, best_i)] = iteration + tabu_tenure

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No improving move found
                no_improve_count += 1

            iteration += 1

            # Clean up old tabu entries to save memory
            if iteration % 100 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            if no_improve_count > max_no_improve:
                break

        return best_p, best_cost

    # Hungarian Re-optimization using Marginal Costs
    def hungarian_reoptimize(current_p):
        # B_perm is B with columns permuted by current_p
        # B_perm[k, j] = B[k, current_p[j]]
        B_perm = b_float[:, current_p]
        
        # M[i, k] = sum_{j != i} (a[i, j] + a[j, i]) * B[k, current_p[j]]
        # M_raw = (A + A.T) @ B_perm.T
        # M_raw[i, k] = sum_j (A+A.T)[i, j] B_perm[k, j]
        
        A_sym_part = a_float + a_float.T
        M_raw = A_sym_part @ B_perm.T
        
        # Subtract the j=i term: (A+A.T)[i, i] * B_perm[k, i] = 2 * A[i, i] * B[k, current_p[i]]
        # We need to subtract this for each k.
        # diag_A = A.diagonal()
        # subtract_matrix = 2 * diag_A[:, None] * b_float[:, current_p].T
        # M = M_raw - subtract_matrix
        
        diag_A = np.diag(a_float)
        # b_float[:, current_p].T has shape (n, n) where row k is B[:, current_p].T[k] = B[current_p[k], :]^T? No.
        # B_perm.T[k, j] = B_perm[j, k] = B[current_p[j], k].
        # We want B[k, current_p[i]] for the subtraction term at index (i, k).
        # Let's construct the subtraction matrix S where S[i, k] = 2 * A[i, i] * B[k, current_p[i]]
        # S = 2 * diag_A[:, None] * B_sub.T
        # Where B_sub = b_float[:, current_p]
        # B_sub.T[k, i] = B_sub[i, k] = B[current_p[i], k] = B[k, current_p[i]] (since B is symmetric)
        # So S[i, k] = 2 * A[i, i] * B_sub.T[k, i]? No.
        # S[i, k] = 2 * A[i, i] * B[k, current_p[i]]
        # Let V = b_float[:, current_p] # V[j, i] = B[j, current_p[i]]
        # We want S[i, k] = 2 * A[i, i] * V[k, i]
        # S = 2 * diag_A[:, None] * V.T
        # V.T[i, k] = V[k, i] = B[k, current_p[i]]
        # So S[i, k] = 2 * A[i, i] * V.T[i, k]
        
        V = b_float[:, current_p]
        S = 2.0 * diag_A[:, None] * V.T
        
        M = M_raw - S
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(M)
        new_p = col_ind.copy()
        
        # Calculate cost of new_p
        new_cost = calculate_cost(new_p)
        
        return new_p, new_cost

    # Main Loop
    best_p = None
    best_cost = float('inf')

    # 1. Initial Solution: Hungarian on linearized cost
    C_init = a_float @ b_float.T
    row_ind, col_ind = linear_sum_assignment(C_init)
    p_init = col_ind.copy()
    
    # 2. Polish with Tabu
    p_init, cost_init = tabu_search(p_init, time_limit - 20.0)
    best_p = p_init
    best_cost = cost_init

    # 3. Iterative Hungarian Re-optimization and Tabu Polish
    # We alternate between Hungarian steps (global reset) and Tabu (local polish)
    # Hungarian is fast for n=256, so we can do several cycles.
    
    while time.time() - start_time < time_limit - 5.0:
        # Step A: Hungarian Re-optimization based on current best
        # This tries to find a globally better assignment given the current "context"
        p_hung, cost_hung = hungarian_reoptimize(best_p)
        
        if cost_hung < best_cost:
            best_cost = cost_hung
            best_p = p_hung
        else:
            # If Hungarian didn't improve, try a perturbed Hungarian
            # Or just proceed to Tabu on the current best
            pass
            
        # Step B: Polish the result of Hungarian (or current best) with Tabu
        # Allocate remaining time carefully
        time_elapsed = time.time() - start_time
        remaining = time_limit - 5.0 - time_elapsed
        
        if remaining < 2.0:
            break
            
        # Give Tabu a portion of the remaining time
        # If we just did a big jump, give it more time. Otherwise, less.
        tabu_time = min(remaining * 0.5, 10.0) # Max 10s per tabu phase
        
        p_polished, cost_polished = tabu_search(best_p, time.time() + tabu_time)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished

    # Final safety check
    if best_p is None or sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()

    return best_p.tolist()
# EVOLVE-BLOCK-END
