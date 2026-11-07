"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: leave 10s safety margin before the 120s limit
    deadline = time.time() + 110.0
    
    # Greedy initialization: largest demand first, cheapest feasible facility
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.float64)
    order = np.argsort(-demands)
    for i in order:
        # Find cheapest facility with remaining capacity
        costs = assign_cost[i].copy()
        for f in np.argsort(costs):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    
    # Local search: first-improvement moves
    # We try to reassign each customer to a different facility if it reduces total cost
    # and respects capacity constraints.
    
    def compute_total_cost(res):
        """Compute total cost of an assignment."""
        total = 0.0
        # Assignment costs
        total += np.sum(assign_cost[np.arange(n_cust), res])
        # Opening costs for used facilities
        used = np.unique(res)
        total += np.sum(open_cost[used])
        return total
    
    best_result = result.copy()
    best_cost = compute_total_cost(best_result)
    
    # Track load for quick feasibility checks
    current_load = load.copy()
    
    # Multiple passes of local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order for diversity
        for i in np.random.permutation(n_cust):
            if time.time() > deadline:
                break
            current_fac = result[i]
            # Try all other facilities
            for f in np.arange(n_fac):
                if f == current_fac:
                    continue
                # Check capacity
                if current_load[f] + demands[i] > capacities[f]:
                    continue
                # Compute cost change
                # Remove i from current_fac, add to f
                # Assignment cost change
                old_assign = assign_cost[i, current_fac]
                new_assign = assign_cost[i, f]
                assign_delta = new_assign - old_assign
                
                # Opening cost change:
                # If current_fac becomes empty, we save open_cost[current_fac]
                # If f was empty, we pay open_cost[f]
                old_load_cf = current_load[current_fac]
                new_load_f = current_load[f]
                
                open_delta = 0.0
                if old_load_cf == demands[i]:  # current_fac becomes empty
                    open_delta -= open_cost[current_fac]
                if new_load_f == 0:  # f was empty
                    open_delta += open_cost[f]
                
                total_delta = assign_delta + open_delta
                
                if total_delta < -1e-9:  # Strict improvement
                    # Apply move
                    current_load[current_fac] -= demands[i]
                    current_load[f] += demands[i]
                    result[i] = f
                    improved = True
                    break  # Move to next customer
    
    # Verify validity of best_result
    final_load = np.zeros(n_fac, dtype=np.float64)
    for i in range(n_cust):
        final_load[result[i]] += demands[i]
    
    valid = True
    for f in range(n_fac):
        if final_load[f] > capacities[f] + 1e-9:
            valid = False
            break
        if result[i] < 0 or result[i] >= n_fac:
            valid = False
            break
    
    if not valid:
        # Fallback to simple greedy
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.float64)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    
    return result
# EVOLVE-BLOCK-END
