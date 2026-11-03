"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initial Solution: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    # Calculate initial cost
    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    best_p = p.copy()
    best_cost = calculate_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # Precompute symmetric parts for speed if matrices are symmetric (QAPLIB standard)
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if a_sym and b_sym:
        # Use efficient delta calculation
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
        # Tabu Search Structures
        tabu_list = {}
        tabu_tenure = 150
        iteration = 0
        
        # Frequency memory for diversification
        freq_i = np.zeros(n, dtype=np.int64)
        freq_j = np.zeros(n, dtype=np.int64)
        
        # Helper to calculate delta for swapping i and j
        def calc_delta(i, j, perm):
            u = perm[i]
            v = perm[j]
            
            delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
            
            diff_A = a[i, :] - a[j, :]
            diff_B = b[v, perm] - b[u, perm]
            
            total_dot = np.dot(diff_A, diff_B)
            
            exclude_i = diff_A[i] * diff_B[i]
            exclude_j = diff_A[j] * diff_B[j]
            
            off_diag_sum = total_dot - exclude_i - exclude_j
            delta += 2 * off_diag_sum
            
            return int(delta)
        
        # Hungarian-based diversification
        def hungarian_kick():
            nonlocal current_p, current_cost, best_p, best_cost
            # Compute marginal cost matrix for assignment
            # Cost of assigning facility i to location k is sum_j a[i,j] * b[k, p[j]]
            # We want to minimize sum_i c[i, p[i]]
            # c[i, k] = sum_j a[i,j] * b[k, p[j]]
            
            # Efficiently compute cost matrix C where C[i, k] = sum_j a[i, j] * b[k, p[j]]
            # Note: b[k, p[j]] is the distance from location k to the location currently assigned to facility j.
            # Let B_p be the matrix b with columns permuted by p? No, b[k, p[j]] is element (k, p[j]) of b.
            # Let's use matrix multiplication.
            # We have a (n x n) and b (n x n).
            # We want C[i, k] = sum_j a[i, j] * b[k, p[j]].
            # Let B_perm be such that B_perm[k, j] = b[k, p[j]].
            # This is equivalent to b[:, p] (column indexing).
            B_perm = b[:, current_p]
            # C = a @ B_perm.T ?
            # (a @ B_perm.T)[i, k] = sum_j a[i, j] * B_perm.T[j, k] = sum_j a[i, j] * B_perm[k, j] = sum_j a[i, j] * b[k, p[j]].
            # Yes.
            C = a @ B_perm.T
            
            # Solve assignment problem
            # linear_sum_assignment minimizes sum_i C[i, row_i]
            # Here we want to assign facilities (rows of C) to locations (cols of C).
            # So we solve min sum_i C[i, loc_i].
            row_ind, col_ind = linear_sum_assignment(C)
            
            # row_ind is just 0..n-1 usually, col_ind is the permutation
            new_p = col_ind.copy()
            new_cost = calculate_cost(new_p)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p.copy()
            
            # Always accept the new solution to move to a new basin?
            # Or only if it's better than current? 
            # Usually for diversification, we accept even if worse, but we track best.
            # However, if it's much worse, we might waste time.
            # Let's accept it as the new current state to explore the new basin.
            current_p = new_p
            current_cost = new_cost
            
            # Clear tabu list to avoid bias from previous basin
            tabu_list.clear()

        # Main Loop
        while time.time() - start_time < time_limit:
            improved = False
            best_delta_in_pass = 0
            best_move_i = -1
            best_move_j = -1
            
            # Iterate through pairs
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                u = current_p[i]
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    v = current_p[j]
                    
                    key1 = (i, j)
                    key2 = (j, i)
                    is_tabu = (key1 in tabu_list and tabu_list[key1] > iteration) or \
                              (key2 in tabu_list and tabu_list[key2] > iteration)
                    
                    delta = calc_delta(i, j, current_p)
                    
                    # Aspiration: accept if it improves global best even if tabu
                    if delta < 0 or (is_tabu and current_cost + delta < best_cost):
                        current_p[i], current_p[j] = v, u
                        current_cost += delta
                        
                        tabu_list[(i, j)] = iteration + tabu_tenure
                        tabu_list[(j, i)] = iteration + tabu_tenure
                        
                        freq_i[i] += 1
                        freq_i[j] += 1
                        freq_j[i] += 1
                        freq_j[j] += 1
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                        
                        improved = True
                        break 
                
                if improved:
                    break
                    
            iteration += 1
            
            if not improved:
                # Local optimum reached
                if time.time() - start_time > time_limit - 10.0:
                    break
                    
                # Diversification: Hungarian Kick
                # This is a global move, so we do it when stuck.
                hungarian_kick()
                
                # After kick, do a few quick 2-opt improvements to settle in the new basin
                # This prevents the next hungarian call from being too far away or from being stuck in a bad spot
                for _ in range(10):
                    if time.time() - start_time > time_limit - 5.0:
                        break
                    found_move = False
                    # Randomize a few pairs to check
                    for _ in range(50):
                        i = random.randint(0, n-1)
                        j = random.randint(0, n-1)
                        if i == j: continue
                        delta = calc_delta(i, j, current_p)
                        if delta < 0:
                            current_p[i], current_p[j] = current_p[j], current_p[i]
                            current_cost += delta
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_p = current_p.copy()
                            found_move = True
                            break
                    if not found_move:
                        break

    # Final safety check: ensure best_p is a valid permutation
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
