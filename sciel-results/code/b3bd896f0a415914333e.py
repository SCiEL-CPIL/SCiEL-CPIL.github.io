"""Capacitated Facility Location Solver using Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost.
    
    Args:
        demands: Array of customer demands.
        capacities: Array of facility capacities.
        open_cost: Array of facility opening costs.
        assign_cost: 2D array of assignment costs (customer, facility).
        
    Returns:
        Array of facility indices assigned to each customer.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: 150s limit, aim for 140s
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Initialization: Greedy Largest Demand First
    # Sort customers by demand descending
    order = np.argsort(-demands)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=int)
    
    for i in order:
        # Find cheapest feasible facility
        # We can vectorize this check
        feasible = (load + demands[i]) <= capacities
        costs = assign_cost[i].copy()
        costs[~feasible] = np.inf
        best_f = np.argmin(costs)
        result[i] = best_f
        load[best_f] += demands[i]

    # Helper to calculate current total cost
    def calc_cost(res, ld):
        open_facs = np.where(ld > 0)[0]
        open_c = np.sum(open_cost[open_facs])
        assign_c = np.sum(assign_cost[np.arange(n_cust), res])
        return open_c + assign_c

    current_cost = calc_cost(result, load)
    
    # 2. Local Search
    # We alternate between Relocate and Swap moves
    # We use "First Improvement" or "Best Improvement"? 
    # Given small N, Best Improvement per pass is good.
    # But we can just loop until no improvement.
    
    max_iterations = 10000 # Safety cap, though time is the real limit
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        improved = False
        
        # --- Relocate Pass ---
        # Try moving each customer to a better facility
        # To ensure we don't miss improvements due to order, we can shuffle or just iterate.
        # Iterating 0..N is fine.
        for i in range(n_cust):
            old_f = result[i]
            d_i = demands[i]
            
            # Vectorized delta calculation
            # Base delta in assignment cost
            delta_assign = assign_cost[i, :] - assign_cost[i, old_f]
            
            # Delta in open cost
            delta_open = np.zeros(n_fac)
            
            # If old facility becomes empty
            if load[old_f] == d_i:
                delta_open[old_f] -= open_cost[old_f]
                
            # If new facility was empty
            empty_facs = (load == 0)
            delta_open[empty_facs] += open_cost[empty_facs]
            
            total_delta = delta_assign + delta_open
            
            # Mask infeasible facilities (capacity)
            feasible_mask = (load + d_i) <= capacities
            # Also exclude old facility (no-op)
            feasible_mask[old_f] = False
            
            if not np.any(feasible_mask):
                continue
                
            # Find best feasible facility
            valid_deltas = np.where(feasible_mask, total_delta, np.inf)
            best_f = np.argmin(valid_deltas)
            best_delta = valid_deltas[best_f]
            
            if best_delta < 0:
                # Apply move
                load[old_f] -= d_i
                load[best_f] += d_i
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        # --- Swap Pass ---
        # Try swapping pairs of customers in different facilities
        # O(N^2) is 10,000 checks. Very fast.
        for i in range(n_cust):
            f_i = result[i]
            d_i = demands[i]
            for j in range(i + 1, n_cust):
                f_j = result[j]
                if f_i == f_j:
                    continue
                
                d_j = demands[j]
                
                # Check capacity feasibility
                # New load for f_i: load[f_i] - d_i + d_j
                # New load for f_j: load[f_j] - d_j + d_i
                new_load_i = load[f_i] - d_i + d_j
                new_load_j = load[f_j] - d_j + d_i
                
                if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                    continue
                
                # Calculate cost delta
                # Assignment cost change
                delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                               (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost change
                # As analyzed, swaps between distinct facilities don't change which facilities are open
                # unless a facility becomes empty.
                # Can f_i become empty?
                # load[f_i] - d_i + d_j == 0
                # Since d_j > 0, load[f_i] - d_i must be negative? No, load[f_i] >= d_i.
                # So load[f_i] - d_i >= 0.
                # If load[f_i] == d_i, then new load is d_j > 0. So f_i stays open.
                # Similarly f_j stays open.
                # So delta_open = 0.
                
                total_delta = delta_assign
                
                if total_delta < 0:
                    # Apply swap
                    load[f_i] = new_load_i
                    load[f_j] = new_load_j
                    result[i], result[j] = f_j, f_i
                    current_cost += total_delta
                    improved = True
        
        if not improved:
            break
            
        iteration += 1
        
        # Time check inside loop for safety
        if time.time() > deadline:
            break

    return result
# EVOLVE-BLOCK-END
