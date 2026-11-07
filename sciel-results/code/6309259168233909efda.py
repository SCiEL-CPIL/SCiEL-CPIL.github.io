"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(result) != n_cust:
        return np.inf
    for i in range(n_cust):
        if result[i] < 0 or result[i] >= n_fac:
            return np.inf
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    for f in range(n_fac):
        if load[f] > capacities[f]:
            return np.inf
    total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            total += open_cost[f]
    for i in range(n_cust):
        total += assign_cost[i, result[i]]
    return total


def _greedy_init(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        assigned = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                assigned = True
                break
        if not assigned:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            f = np.argmax(rem)
            load[f] += demands[i]
            result[i] = f
    return result


def _local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0):
    """Local search: move customers to reduce cost while respecting capacity."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            current_f = result[i]
            
            # Get all facilities sorted by assign_cost for customer i
            for f in np.argsort(assign_cost[i]):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                
                # Calculate delta cost
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Check if current facility becomes empty
                if load[current_f] == demands[i]:
                    delta -= open_cost[current_f]
                
                # Check if new facility was empty
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < -1e-9:
                    # Accept the move
                    load[current_f] -= demands[i]
                    load[f] += demands[i]
                    result[i] = f
                    best_cost += delta
                    improved = True
                    break  # Move this customer to the best feasible facility and break
    
    return result


def _swap_local_search(demands, capacities, open_cost, assign_cost, result, time_limit=110.0):
    """Local search: swap customers between facilities to reduce cost."""
    start_time = time.time()
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return result
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            fi = result[i]
            for j in range(i + 1, n_cust):
                if time.time() - start_time > time_limit:
                    break
                fj = result[j]
                if fi == fj:
                    continue
                
                # Check capacity feasibility after swap
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                
                # Calculate delta cost
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes: if a facility becomes empty or was empty
                # After swap, fi has load[fi] - demands[i] + demands[j]
                # After swap, fj has load[fj] - demands[j] + demands[i]
                # Before: fi had load[fi], fj had load[fj]
                
                # If fi was non-empty and becomes empty after swap
                if load[fi] > 0 and (load[fi] - demands[i] + demands[j] == 0):
                    delta -= open_cost[fi]
                # If fj was non-empty and becomes empty after swap
                if load[fj] > 0 and (load[fj] - demands[j] + demands[i] == 0):
                    delta -= open_cost[fj]
                # If fi was empty and becomes non-empty after swap (impossible since load[fi]>=demands[i]>0)
                # If fj was empty and becomes non-empty after swap (impossible since load[fj]>=demands[j]>0)
                
                if delta < -1e-9:
                    # Accept the swap
                    load[fi] = load[fi] - demands[i] + demands[j]
                    load[fj] = load[fj] - demands[j] + demands[i]
                    result[i] = fj
                    result[j] = fi
                    best_cost += delta
                    improved = True
                    break  # Break inner loop, continue with next i
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities using greedy initialization and local search."""
    start_time = time.time()
    
    # Greedy initialization
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    # First pass: move local search
    time_limit_move = max(1.0, 50.0 - (time.time() - start_time))
    result = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit_move)
    
    # Second pass: swap local search
    elapsed = time.time() - start_time
    time_limit_swap = max(1.0, 100.0 - elapsed)
    result = _swap_local_search(demands, capacities, open_cost, assign_cost, result, time_limit_swap)
    
    # Third pass: another move local search with remaining time
    elapsed = time.time() - start_time
    time_limit_move2 = max(1.0, 110.0 - elapsed)
    result = _local_search(demands, capacities, open_cost, assign_cost, result, time_limit_move2)
    
    return result
# EVOLVE-BLOCK-END
