"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # Helper to compute full cost
    def compute_cost(p):
        B = b[np.ix_(p, p)]
        return int(np.sum(a * B))
        
    # Helper to compute delta cost for swapping i and j
    def delta_cost(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        b_p_v = b[p, v]
        b_p_u = b[p, u]
        
        diff_v_u_out = b_v_p - b_u_p
        diff_v_u_in = b_p_v - b_p_u
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        s1 = np.dot(a[i, mask], diff_v_u_out[mask])
        s2 = np.dot(a[mask, i], diff_v_u_in[mask])
        s3 = np.dot(a[j, mask], -diff_v_u_out[mask])
        s4 = np.dot(a[mask, j], -diff_v_u_in[mask])
        
        sum_part = s1 + s2 + s3 + s4
        
        cross_delta = a[i, j] * b[v, u] + a[j, i] * b[u, v] - (a[i, j] * b[u, v] + a[j, i] * b[v, u])
        
        return sum_part + cross_delta

    # Initial Solution: Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    facility_order = np.argsort(-a_row_sums, kind='stable')
    location_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu Search Parameters
    tabu_tenure = 20  # Length of tabu list
    max_iter = 100000 # Safety cap, but time limit is primary
    
    start_time = time.time()
    time_limit = 115.0
    
    rng = np.random.default_rng(42)
    
    # Tabu list stores tuples (i, j) or a set of forbidden swaps
    # Using a dictionary to store expiration times for each swap (i, j)
    tabu = {}
    
    # To speed up, we can precompute some row sums of A? 
    # The delta cost function is O(n). For n=150, it's fast enough.
    # We will use First-Improvement strategy to find a move quickly.
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # Find a move
        best_delta = 0 # We accept moves with delta < 0, or tabu-breaking moves
        best_i, best_j = -1, -1
        is_tabu_best = False
        
        # To avoid O(n^2) full scan if it's too slow, we can sample or use full scan.
        # n=150 -> 11175 pairs. Delta calc is ~150 ops. 11175 * 150 ~ 1.6M ops.
        # This is fast in NumPy/Python loop? Python loop over 11k items is ~1ms.
        # So full scan is feasible.
        
        found_move = False
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = delta_cost(i, j, p)
                
                # Check tabu status
                # Normalize key to (min, max)
                key = (i, j)
                is_tabu = key in tabu and tabu[key] > iteration
                
                if d < 0:
                    if not is_tabu:
                        # Non-tabu improving move
                        if d < best_delta:
                            best_delta = d
                            best_i, best_j = i, j
                            is_tabu_best = False
                            found_move = True
                    else:
                        # Tabu improving move - only accept if aspiration (better than best_cost)
                        # We don't have current_cost + d < best_cost check here easily without current_cost
                        # We can store current_cost
                        pass # Simplified: ignore tabu improving moves unless aspiration
                else:
                    # Non-improving move
                    if is_tabu:
                        continue
                    # If we haven't found an improving move, consider non-improving non-tabu moves?
                    # Standard TS: Choose best non-tabu move, even if it increases cost.
                    if not found_move:
                        if d < best_delta:
                            best_delta = d
                            best_i, best_j = i, j
                            is_tabu_best = False
                            found_move = True
        
        if not found_move:
            # No non-tabu move found? This shouldn't happen if tabu list is small relative to n^2
            # Or all moves are tabu. In that case, pick random non-tabu or break.
            # For robustness, if no move found, pick a random swap
            best_i = rng.integers(n)
            best_j = rng.integers(n)
            while best_i == best_j:
                best_j = rng.integers(n)
            best_delta = delta_cost(best_i, best_j, p)
            is_tabu_best = True # Force accept
            
        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu List
        key = (min(best_i, best_j), max(best_i, best_j))
        tabu[key] = iteration + tabu_tenure
        
        iteration += 1
        
        # Periodically clean up tabu list to save memory (optional, dict is small)
        if iteration % 1000 == 0:
            for k in list(tabu.keys()):
                if tabu[k] <= iteration:
                    del tabu[k]
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
