import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    def _compute_cost(result):
        if len(result) != n_cust:
            return float('inf')
        load = np.zeros(n_fac)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        open_facilities = np.unique(result)
        cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(n_cust), result])
        return cost

    def _greedy_assign(order=None, alpha=1.0):
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

    def _fast_local_search(result, time_limit):
        """
        Fast local search with 1-moves and 2-swaps.
        """
        n_cust_local = n_cust
        n_fac_local = n_fac
        current = result.astype(int).copy()
        load = np.zeros(n_fac_local)
        fac_count = np.zeros(n_fac_local, dtype=int)
        
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        # Precompute current cost components
        current_cost = _compute_cost(current)
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        # Pre-sort customers by demand descending for potential tie-breaking or just use random order?
        # Using random order in local search can help escape local optima slightly, but fixed is faster.
        # Let's stick to fixed order for speed, maybe randomize in outer loops if time permits?
        # Actually, randomizing customer order in each pass of local search is cheap and helps.
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (1-move)
            # Iterate customers in random order to reduce bias
            cust_order = np.random.permutation(n_cust_local)
            
            for i in cust_order:
                if time.time() - start > time_limit:
                    return best_result
                
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac_local):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    # Open cost change
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
                    # Apply move
                    load[cur_f] -= d_i
                    load[best_new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[best_new_f] += 1
                    current[i] = best_new_f
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
                        
            # 2. Swap (2-move)
            # Try swapping pairs (i, j) where current[i] != current[j]
            # To speed up, we can just iterate all pairs. n=100, pairs=4950.
            # For each pair, check capacity feasibility and cost delta.
            # We can skip if time is running out.
            
            # Optimization: Only try swaps between customers in different facilities.
            # We can iterate i, then j > i.
            
            # To make it faster, we can group customers by facility.
            # But n_fac is small (20), so checking all pairs is O(N^2 * N_fac) which is 100^2 * 20 = 200k ops.
            # Each op is cheap. 200k * 100 passes = 20M ops. Should be fine in Python if optimized.
            # Let's try to vectorize or optimize the inner loop.
            
            # Actually, let's just do it simply first.
            # If it's too slow, we can restrict swaps to nearby customers or random subsets.
            
            for i in range(n_cust_local):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust_local):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    # Check capacity after swap
                    # f_i loses d_i, gains d_j
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    # f_j loses d_j, gains d_i
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    # Open cost change for swap between different facilities
                    # fac_count[f_i] and fac_count[f_j] remain unchanged (lose 1, gain 1)
                    # So delta open cost is 0
                    
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
                        # Continue to next j? Or break?
                        # Breaking restarts the j loop for this i? No, it just stops looking for swaps for this i.
                        # It's better to continue to find more improvements in the same pass?
                        # Or break to ensure stability?
                        # Let's break to avoid conflicting updates in the same pass for the same i.
                        break
                        
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
    # Run a few times
    for _ in range(3):
        sol_stoch = _greedy_assign(order=np.argsort(-demands), alpha=0.1)
        cost_stoch = _compute_cost(sol_stoch)
        if cost_stoch < best_initial_cost:
            best_initial = sol_stoch
            best_initial_cost = cost_stoch

    # 2. Local Search Polish
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        # Use most of the remaining time for local search
        final_result = _fast_local_search(best_initial, remaining_time - 1)
    else:
        final_result = best_initial
        
    # Final Validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(final_result):
        if f < 0 or f >= n_fac:
            return np.zeros(n_cust, dtype=int) # Invalid fallback
        load_final[f] += demands[i]
        
    if np.any(load_final > capacities + 1e-9):
        # If invalid, return the initial result which was valid (or the best valid found)
        # best_initial was validated by _compute_cost returning inf if invalid, so it should be valid
        return best_initial
        
    return final_result
