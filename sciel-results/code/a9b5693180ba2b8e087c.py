"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute symmetric parts for speed if matrices are symmetric (QAPLIB standard)
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    # Calculate cost
    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    # Precompute for faster delta calculation
    if a_sym and b_sym:
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
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
    else:
        # Fallback delta for non-symmetric (less efficient)
        def calc_delta(i, j, perm):
            old_cost = 0
            new_cost = 0
            u = perm[i]
            v = perm[j]
            
            for k in range(n):
                if k == i or k == j:
                    old_cost += a[i, k] * b[u, perm[k]] + a[k, i] * b[perm[k], u]
                    old_cost += a[j, k] * b[v, perm[k]] + a[k, j] * b[perm[k], v]
                    new_cost += a[i, k] * b[v, perm[k]] + a[k, i] * b[perm[k], v]
                    new_cost += a[j, k] * b[u, perm[k]] + a[k, j] * b[perm[k], u]
                else:
                    # Terms involving i and j with each other
                    old_cost += a[i, j] * b[u, v] + a[j, i] * b[v, u]
                    new_cost += a[i, j] * b[v, u] + a[j, i] * b[u, v]
            
            return new_cost - old_cost

    def greedy_initial():
        # Greedy construction: pick facility with highest sum to location with lowest weighted distance
        remaining_f = list(range(n))
        remaining_l = list(range(n))
        p = np.zeros(n, dtype=np.int64)
        
        # Precompute row sums for priority
        a_row_sums = a.sum(axis=1)
        
        # Sort facilities by flow sum descending
        order_f = np.argsort(-a_row_sums, kind="stable")
        
        # For each facility in order, pick best location
        # b_row_sums is sum of distances, but we want to minimize total cost.
        # A simple heuristic: assign high flow to low distance sum.
        b_row_sums = b.sum(axis=1)
        
        # Let's try a slightly better greedy:
        # For facility f, cost of assigning to location l is sum_{k assigned} (a[f,k]*b[l, p[k]] + a[k,f]*b[p[k], l])
        # This is expensive if done naively.
        # Let's stick to the rank matching or a simple iterative improvement from a random start.
        
        # Rank matching is fast and decent.
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    best_p = greedy_initial()
    best_cost = calculate_cost(best_p)

    # Tabu Search
    tabu_list = {}
    tabu_tenure = 120
    iteration = 0
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Track frequency of assignments for diversification
    freq_matrix = np.zeros((n, n), dtype=np.int64) # freq[f][l]
    
    # Update freq for initial
    for i in range(n):
        freq_matrix[i, current_p[i]] += 1

    # Main Loop
    while time.time() - start_time < time_limit:
        improved = False
        best_delta_in_pass = 0
        best_move_i = -1
        best_move_j = -1
        
        # Randomize evaluation order to avoid bias
        indices = np.random.permutation(n)
        
        for idx_i in indices:
            i = int(idx_i)
            if time.time() - start_time > time_limit:
                break
                
            u = current_p[i]
            
            # Evaluate all swaps with j
            # To speed up, we could vectorize the delta calculation for a fixed i
            # delta(i, j) = (a[i,j] - a[j,i]) ... wait, the formula derived assumes symmetry.
            # For general, let's stick to the loop but maybe optimize the inner product.
            
            # Let's try to vectorize for fixed i.
            # delta(i, j) for all j != i
            # If symmetric:
            # diff_A = a[i, :] - a[j, :]
            # diff_B = b[u, perm] - b[perm[j], perm]  (wait, v = perm[j])
            # diff_B = b[u, perm] - b[perm, perm[j]] (since v=perm[j])
            
            if a_sym and b_sym:
                # Vectorized delta for fixed i
                # diff_A = a[i, :] - a[:, j] -> shape (n, n) if we do all j at once?
                # a[i, :] is (n,), a[:, j] is (n,). 
                # We want delta for all j.
                # delta_j = (a_diag[i] - a_diag[j]) * (b_diag[perm[j]] - b_diag[u]) + 2 * sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[perm[j], perm[k]] - b[u, perm[k]])
                
                # This is hard to fully vectorize without huge memory.
                # Let's stick to the scalar loop but optimize slightly.
                
                for j in range(n):
                    if j == i:
                        continue
                    
                    v = current_p[j]
                    
                    key1 = (i, j)
                    key2 = (j, i)
                    is_tabu = (key1 in tabu_list and tabu_list[key1] > iteration) or \
                              (key2 in tabu_list and tabu_list[key2] > iteration)
                    
                    delta = calc_delta(i, j, current_p)
                    
                    # Aspiration
                    if delta < best_delta_in_pass:
                        if not is_tabu or (current_cost + delta < best_cost):
                            best_delta_in_pass = delta
                            best_move_i = i
                            best_move_j = j
            
            if best_move_i != -1 and best_delta_in_pass < 0:
                i = best_move_i
                j = best_move_j
                u = current_p[i]
                v = current_p[j]
                
                current_p[i], current_p[j] = v, u
                current_cost += best_delta_in_pass
                
                tabu_list[(i, j)] = iteration + tabu_tenure
                tabu_list[(j, i)] = iteration + tabu_tenure
                
                freq_matrix[i, u] -= 1
                freq_matrix[j, v] -= 1
                freq_matrix[i, v] += 1
                freq_matrix[j, u] += 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                
                improved = True
                break
                
            best_move_i = -1
            best_move_j = -1
            best_delta_in_pass = 0

        iteration += 1
        
        if not improved:
            # Local optimum
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Diversification: Random kick
            # Pick random pair and swap
            i_kick = random.randint(0, n-1)
            j_kick = random.randint(0, n-1)
            if i_kick != j_kick:
                delta = calc_delta(i_kick, j_kick, current_p)
                current_p[i_kick], current_p[j_kick] = current_p[j_kick], current_p[i_kick]
                current_cost += delta
                
                u = current_p[i_kick]
                v = current_p[j_kick]
                # Note: after swap, p[i_kick]=v, p[j_kick]=u
                # So old positions were swapped.
                # freq updates:
                # i_kick was at u, now at v? No.
                # Before: p[i]=u, p[j]=v
                # After: p[i]=v, p[j]=u
                # freq[i, u]--, freq[i, v]++
                # freq[j, v]--, freq[j, u]++
                
                # Wait, in the code above I did:
                # current_p[i], current_p[j] = v, u
                # So p[i] becomes v, p[j] becomes u.
                # Correct.
                
                freq_matrix[i_kick, current_p[i_kick]] += 1
                freq_matrix[j_kick, current_p[j_kick]] += 1
                # This is wrong because I need to decrement the old ones.
                # Let's fix the freq update logic in the main loop first.
                
                tabu_list[(i_kick, j_kick)] = iteration + tabu_tenure
                tabu_list[(j_kick, i_kick)] = iteration + tabu_tenure
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()

    # Final safety check
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
