"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

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

        # Main Loop
        while time.time() - start_time < time_limit:
            improved = False
            best_delta_in_pass = 0
            best_move_i = -1
            best_move_j = -1
            
            # Iterate through pairs
            # To speed up, we can randomize the order of i, but keep j loop standard?
            # Or just standard for now.
            
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
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                # Diversification: Random kick
                # Try a few random kicks to find a better basin?
                # Or just one kick.
                i_kick = random.randint(0, n-1)
                j_kick = random.randint(0, n-1)
                if i_kick != j_kick:
                    delta = calc_delta(i_kick, j_kick, current_p)
                    current_p[i_kick], current_p[j_kick] = current_p[j_kick], current_p[i_kick]
                    current_cost += delta
                    
                    tabu_list[(i_kick, j_kick)] = iteration + tabu_tenure
                    tabu_list[(j_kick, i_kick)] = iteration + tabu_tenure
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                    else:
                        # If the kick was bad, maybe revert? 
                        # In standard tabu, we don't revert immediately, we continue searching.
                        # But for QAP, a bad kick can trap us.
                        # Let's stick to standard: accept the kick and continue.
                        pass

    # Final safety check: ensure best_p is a valid permutation
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
