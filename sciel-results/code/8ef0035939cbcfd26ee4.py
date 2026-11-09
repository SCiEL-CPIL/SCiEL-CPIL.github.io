"""Capacitated facility location: Portfolio of greedy seeds + Best Improvement Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_result = None
    best_cost = float('inf')

    def _total_cost(result, demands, capacities, open_cost, assign_cost):
        """Compute total cost: open costs + assignment costs."""
        n_fac = len(open_cost)
        load = np.zeros(n_fac)
        for c, f in enumerate(result):
            load[f] += demands[c]
        used = load > 0
        open_sum = open_cost[used].sum()
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def _greedy_seed(demands, capacities, open_cost, assign_cost, strategy='eff'):
        n_cust = len(demands)
        n_fac = len(capacities)
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac, dtype=float)

        # Determine order
        if strategy == 'eff':
            amortized = open_cost / np.maximum(capacities, 1.0)
            eff_cost = assign_cost + np.outer(demands, amortized)
            total_eff = eff_cost.sum(axis=1)
            order = np.lexsort((total_eff, -demands))
            cost_matrix_for_sort = eff_cost
        elif strategy == 'raw':
            order = np.argsort(-demands)
            cost_matrix_for_sort = assign_cost
        elif strategy == 'random':
            order = np.random.permutation(n_cust)
            cost_matrix_for_sort = assign_cost
        elif strategy == 'small_first':
            order = np.argsort(demands)
            cost_matrix_for_sort = assign_cost
        else:
            order = np.argsort(-demands)
            cost_matrix_for_sort = assign_cost

        for i in order:
            # Sort facilities by the relevant cost metric
            fac_order = np.argsort(cost_matrix_for_sort[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: cheapest feasible by raw assign_cost
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
        
        # If any unassigned, do a final pass with any feasible
        unassigned = np.where(result < 0)[0]
        for i in unassigned:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
            if result[i] < 0:
                # Should not happen if total capacity > total demand, but just in case
                # Assign to facility with most remaining capacity
                remaining = capacities - load
                f = np.argmax(remaining)
                result[i] = f
                load[f] += demands[i]

        return result

    def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
        """Best Improvement Local Search with Relocate and Swap moves."""
        n_cust = len(demands)
        n_fac = len(capacities)
        result = result.copy()
        load = np.zeros(n_fac)
        for c, f in enumerate(result):
            load[f] += demands[c]
        
        current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
        
        # Precompute sorted facilities by assign_cost for quick access if needed, 
        # but Best Improvement requires checking all feasible anyway for delta.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # --- Phase 1: Best Relocate ---
            # For each customer, find the move that gives the best (most negative) delta
            best_move_delta = 0.0
            best_move_c = -1
            best_move_f = -1
            
            for c in range(n_cust):
                old_f = result[c]
                d_c = demands[c]
                base_assign_cost = assign_cost[c, old_f]
                
                # Check all other facilities
                for f in range(n_fac):
                    if f == old_f:
                        continue
                    if load[f] + d_c <= capacities[f]:
                        # Calculate delta
                        delta = assign_cost[c, f] - base_assign_cost
                        
                        # Open cost changes
                        new_load_old = load[old_f] - d_c
                        if new_load_old <= 0:
                            delta -= open_cost[old_f]
                        
                        if load[f] <= 0:
                            delta += open_cost[f]
                            
                        if delta < best_move_delta:
                            best_move_delta = delta
                            best_move_c = c
                            best_move_f = f
                            
            if best_move_delta < -1e-9 and best_move_c != -1:
                c = best_move_c
                f = best_move_f
                old_f = result[c]
                load[old_f] -= demands[c]
                load[f] += demands[c]
                result[c] = f
                current_cost += best_move_delta
                improved = True
                continue # Restart loop to re-evaluate all moves with new state

            # --- Phase 2: Best Swap ---
            # For each pair of customers, find the swap that gives the best delta
            best_swap_delta = 0.0
            best_swap_c1 = -1
            best_swap_c2 = -1
            
            for c1 in range(n_cust):
                f1 = result[c1]
                d1 = demands[c1]
                ac1_f1 = assign_cost[c1, f1]
                
                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]
                    
                    # Check capacity after swap
                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1
                    
                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue
                    
                    # Calculate delta
                    # Assign cost change
                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (ac1_f1 + assign_cost[c2, f2])
                    
                    # Open cost change
                    # f1: was used (load>0), still used (new_load>0) -> no change unless load becomes 0? 
                    # Since we are swapping, and d1,d2 > 0, and they were assigned, 
                    # the facilities remain used as long as they had other customers or the swap keeps them non-empty.
                    # Wait, if f1 only had c1, and we move c1 out, but c2 comes in, it's still used.
                    # So open cost delta is 0 for swaps where both facilities remain non-empty.
                    # Can a facility become empty?
                    # new_load_f1 = load[f1] - d1 + d2. 
                    # If load[f1] was exactly d1 (only c1), then new_load_f1 = d2 > 0.
                    # So facilities involved in a valid swap never become empty.
                    # Thus, delta_open = 0.
                    
                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_swap_c1 = c1
                        best_swap_c2 = c2
                        
            if best_swap_delta < -1e-9 and best_swap_c1 != -1:
                c1 = best_swap_c1
                c2 = best_swap_c2
                f1 = result[c1]
                f2 = result[c2]
                
                # Apply swap
                load[f1] = load[f1] - demands[c1] + demands[c2]
                load[f2] = load[f2] - demands[c2] + demands[c1]
                result[c1] = f2
                result[c2] = f1
                current_cost += best_swap_delta
                improved = True
                continue # Restart loop

        return result, current_cost

    # Portfolio of strategies
    strategies = ['eff', 'raw', 'random', 'small_first']
    
    for strat in strategies:
        if time.time() >= deadline:
            break
            
        # Generate seed
        seed_result = _greedy_seed(demands, capacities, open_cost, assign_cost, strat)
        
        # Validate seed
        if np.any(seed_result < 0):
            continue
            
        seed_cost = _total_cost(seed_result, demands, capacities, open_cost, assign_cost)
        
        if seed_cost < best_cost:
            best_cost = seed_cost
            best_result = seed_result.copy()
            
        # Local search on this seed
        if time.time() < deadline - 10: # Reserve time for other strategies
            ls_result, ls_cost = _local_search(seed_result, demands, capacities, open_cost, assign_cost, deadline)
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_result = ls_result

    # Final fallback if something went wrong
    if best_result is None or np.any(best_result < 0):
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost, 'eff')
        
    # Final validation check
    if np.any(best_result < 0) or np.any(best_result >= n_fac):
        # Force valid assignment
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    best_result[i] = f
                    load[f] += demands[i]
                    break
            if best_result[i] == 0 and load[0] + demands[i] > capacities[0]:
                # Try to find any feasible
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        best_result[i] = f
                        load[f] += demands[i]
                        break

    return best_result.astype(int)
# EVOLVE-BLOCK-END
