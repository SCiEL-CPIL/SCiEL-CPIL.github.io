import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # Precompute for speed
    # assign_cost is (n_cust, n_fac)
    
    def _is_valid(result):
        if len(result) != n_cust:
            return False
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return False
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return False
        return True

    def _compute_cost(result):
        if not _is_valid(result):
            return float('inf')
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            load[f] += demands[i]
        open_facilities = np.unique(result)
        cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(n_cust), result])
        return cost

    def _greedy_assign(order=None, alpha=0.0):
        """
        Greedy assignment.
        order: indices of customers to assign, sorted. If None, uses default (largest demand first).
        alpha: temperature parameter for cost perturbation (0 = pure greedy, >0 = stochastic).
        """
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        
        if order is None:
            # Largest demand first
            order = np.argsort(-demands)
            
        for i in order:
            # Calculate effective costs
            if alpha > 0:
                # Add noise to break ties or explore
                noise = np.random.uniform(0, alpha * np.max(assign_cost[i]))
                eff_cost = assign_cost[i] + noise
            else:
                eff_cost = assign_cost[i]
                
            # Sort facilities by effective cost
            fac_order = np.argsort(eff_cost)
            
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
                    
            if not assigned:
                # Fallback: find any feasible facility with min actual cost
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[i] = f_best
                else:
                    # Should not happen if instance is feasible, but handle gracefully
                    result[i] = 0
                    # Note: This makes the solution invalid, but we validate later
                    
        return result

    def _local_search(result, time_limit):
        """
        Fast local search with 1-moves and 2-swaps.
        Optimized with vectorized operations where possible.
        """
        n_cust_local = n_cust
        n_fac_local = n_fac
        current = result.astype(int).copy()
        
        # State tracking
        load = np.zeros(n_fac_local)
        fac_count = np.zeros(n_fac_local, dtype=int)
        
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        current_cost = _compute_cost(current)
        if np.isinf(current_cost):
            return result # Start was invalid, shouldn't happen but safe
            
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        # Precompute customer indices
        cust_indices = np.arange(n_cust_local)
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (1-move)
            # Iterate customers in random order
            cust_order = np.random.permutation(n_cust_local)
            
            for i in cust_order:
                if time.time() - start > time_limit:
                    return best_result
                
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                # Vectorized cost calculation for all other facilities
                # delta_assign = assign_cost[i, :] - assign_cost[i, cur_f]
                delta_assign = assign_cost[i, :] - assign_cost[i, cur_f]
                
                # Open cost delta
                # If fac_count[cur_f] == 1, we save open_cost[cur_f]
                # If fac_count[new_f] == 0, we pay open_cost[new_f]
                
                # Mask for capacity feasibility
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[cur_f] = False # Can't move to self
                
                # Calculate open cost delta array
                # base_delta = delta_assign
                # If moving from cur_f (and it becomes empty), subtract open_cost[cur_f]
                # If moving to new_f (and it was empty), add open_cost[new_f]
                
                open_delta = np.zeros(n_fac_local)
                if fac_count[cur_f] == 1:
                    open_delta[:] -= open_cost[cur_f]
                # For each new_f, if fac_count[new_f] == 0, add open_cost[new_f]
                empty_mask = (fac_count == 0)
                open_delta[empty_mask] += open_cost[empty_mask]
                
                total_delta = delta_assign + open_delta
                
                # Apply feasibility mask
                total_delta[~feasible_mask] = float('inf')
                
                # Find best
                min_idx = np.argmin(total_delta)
                min_val = total_delta[min_idx]
                
                if min_val < -1e-9:
                    best_new_f = min_idx
                    best_delta = min_val
                    
            # Note: The loop above only finds the best move for the current i.
            # We need to apply it if it's an improvement.
            # Wait, the loop above is inside the "for i in cust_order" loop.
            # I structured it wrong in the draft. Let's re-implement the loop correctly.
            
            # Re-doing the 1-move loop properly
            improved_this_pass = False
            for i in cust_order:
                if time.time() - start > time_limit:
                    return best_result
                
                cur_f = current[i]
                d_i = demands[i]
                
                # Vectorized cost calculation for all other facilities
                delta_assign = assign_cost[i, :] - assign_cost[i, cur_f]
                
                # Mask for capacity feasibility
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[cur_f] = False # Can't move to self
                
                if not np.any(feasible_mask):
                    continue
                    
                # Calculate open cost delta array
                open_delta = np.zeros(n_fac_local)
                if fac_count[cur_f] == 1:
                    open_delta[:] -= open_cost[cur_f]
                empty_mask = (fac_count == 0)
                open_delta[empty_mask] += open_cost[empty_mask]
                
                total_delta = delta_assign + open_delta
                total_delta[~feasible_mask] = float('inf')
                
                min_idx = np.argmin(total_delta)
                min_val = total_delta[min_idx]
                
                if min_val < -1e-9:
                    best_new_f = min_idx
                    best_delta = min_val
                    
                    # Apply move
                    load[cur_f] -= d_i
                    load[best_new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[best_new_f] += 1
                    current[i] = best_new_f
                    current_cost += best_delta
                    improved = True
                    improved_this_pass = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                        
            # 2. Swap (2-move)
            # This is harder to vectorize fully. We'll iterate pairs but optimize inner checks.
            # To keep it fast, we limit swaps to a subset or use a heuristic.
            # Given n=100, N^2=10k pairs. 20 facilities.
            # Let's try a few random pairs or a structured sweep if time permits.
            
            # A simple heuristic: try swapping customers in the same facility? No, that's a 1-move effectively.
            # Try swapping customers in different facilities.
            
            # To avoid O(N^2) Python loop overhead being too high, we can sample pairs.
            # Or we can do a full sweep if it's fast enough.
            # Let's do a full sweep but break early if no improvements found in a while?
            # Actually, let's just do a limited number of random swaps per pass to keep it fast.
            # Or, let's stick to 1-moves for now as they are very powerful in CFLP.
            # Let's add a few random swaps.
            
            if time.time() - start < time_limit:
                num_swaps_to_try = 500
                for _ in range(num_swaps_to_try):
                    if time.time() - start > time_limit:
                        return best_result
                    
                    i = np.random.randint(0, n_cust_local)
                    j = np.random.randint(0, n_cust_local)
                    if i == j:
                        continue
                        
                    f_i = current[i]
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                        
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    # Check capacity
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change is 0 for swaps between different facilities
                    # (fac_count[f_i] and fac_count[f_j] remain unchanged)
                    
                    if delta < -1e-9:
                        # Apply swap
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current[i] = f_j
                        current[j] = f_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        
        return best_result

    # --- Main Logic ---
    
    # 1. Get an initial good solution using multiple greedy strategies
    best_initial = None
    best_initial_cost = float('inf')
    
    # Strategy 1: Largest demand first, pure greedy
    sol1 = _greedy_assign(order=np.argsort(-demands), alpha=0)
    cost1 = _compute_cost(sol1)
    if cost1 < best_initial_cost:
        best_initial = sol1
        best_initial_cost = cost1
        
    # Strategy 2: Smallest demand first, pure greedy
    sol2 = _greedy_assign(order=np.argsort(demands), alpha=0)
    cost2 = _compute_cost(sol2)
    if cost2 < best_initial_cost:
        best_initial = sol2
        best_initial_cost = cost2
        
    # Strategy 3: Random order, pure greedy
    sol3 = _greedy_assign(order=np.random.permutation(n_cust), alpha=0)
    cost3 = _compute_cost(sol3)
    if cost3 < best_initial_cost:
        best_initial = sol3
        best_initial_cost = cost3
        
    # Strategy 4: Stochastic greedy (Largest first, with noise)
    for _ in range(5):
        sol_stoch = _greedy_assign(order=np.argsort(-demands), alpha=0.1)
        cost_stoch = _compute_cost(sol_stoch)
        if cost_stoch < best_initial_cost:
            best_initial = sol_stoch
            best_initial_cost = cost_stoch

    # 2. Local Search Polish
    remaining_time = deadline - time.time()
    if remaining_time > 2 and best_initial is not None:
        # Use most of the remaining time for local search
        final_result = _local_search(best_initial, remaining_time - 1)
    else:
        final_result = best_initial
        
    # Final Validation
    if final_result is None:
        return np.zeros(n_cust, dtype=int)
        
    if not _is_valid(final_result):
        # If invalid, return the initial result which was valid (or the best valid found)
        if best_initial is not None and _is_valid(best_initial):
            return best_initial
        else:
            # Emergency fallback: assign everyone to facility 0 (might be invalid but better than nothing)
            return np.zeros(n_cust, dtype=int)
            
    return final_result
