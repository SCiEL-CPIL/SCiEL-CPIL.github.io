"""Improved capacitated facility location solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of a solution."""
    n_fac = len(open_cost)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    # Check capacity feasibility
    if np.any(load > capacities):
        return np.inf
    # Assignment cost
    assign_total = 0.0
    for i, f in enumerate(result):
        assign_total += assign_cost[i, f]
    # Open cost
    open_total = 0.0
    for f in range(n_fac):
        if load[f] > 0:
            open_total += open_cost[f]
    return assign_total + open_total


def _greedy_initial(demands, capacities, open_cost, assign_cost):
    """Greedy: assign each customer to cheapest feasible facility, with facility opening consideration."""
    n_cust = len(demands)
    n_fac = len(open_cost)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    # Sort customers by demand descending (largest first for better packing)
    order = np.argsort(-demands)
    for i in order:
        # For each facility, compute effective cost: assign_cost + marginal open cost if not yet open
        best_f = -1
        best_cost = np.inf
        for f in range(n_fac):
            if load[f] + demands[i] > capacities[f]:
                continue
            # Marginal cost: if facility is open, just assign_cost; if not, assign_cost + open_cost
            marginal = assign_cost[i, f]
            if load[f] == 0:
                marginal += open_cost[f]
            if marginal < best_cost:
                best_cost = marginal
                best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += demands[i]
        else:
            # Fallback: assign to cheapest facility with remaining capacity
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
    # If any customer unassigned, force assign to any feasible facility
    for i in range(n_cust):
        if result[i] < 0:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit=10.0):
    """Local search: try reassigning customers to other facilities, and opening/closing facilities."""
    n_cust = len(demands)
    n_fac = len(open_cost)
    result = result.copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result, current_cost
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        # Try reassigning each customer to a different facility
        for i in range(n_cust):
            if time.time() - start_time > time_limit:
                break
            old_f = result[i]
            old_load_f = load[old_f]
            # Try all other facilities
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                # Compute cost change
                delta = assign_cost[i, new_f] - assign_cost[i, old_f]
                # If old facility becomes empty, save open_cost[old_f]
                if old_load_f == demands[i]:
                    delta -= open_cost[old_f]
                # If new facility was empty, pay open_cost[new_f]
                if load[new_f] == 0:
                    delta += open_cost[new_f]
                
                if delta < -1e-9:
                    # Apply move
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    current_cost += delta
                    improved = True
                    break  # Restart customer loop after a move
    
    return result, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities with capacity constraints."""
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(open_cost)
    
    global_start = time.time()
    time_budget = 130.0  # Leave 20s safety margin for 150s limit
    
    best_result = None
    best_cost = np.inf
    
    # Generate multiple initial solutions and run local search on each
    # Strategy 1: Greedy largest-first with marginal open cost
    result1 = _greedy_initial(demands, capacities, open_cost, assign_cost)
    cost1 = _total_cost(result1, demands, capacities, open_cost, assign_cost)
    if cost1 < best_cost:
        best_cost = cost1
        best_result = result1.copy()
    
    # Strategy 2: Greedy smallest-first
    result2 = _greedy_initial(demands[::-1].copy(), capacities, open_cost, assign_cost[::-1].copy())
    # Reverse back
    result2 = result2[::-1].copy()
    cost2 = _total_cost(result2, demands, capacities, open_cost, assign_cost)
    if cost2 < best_cost:
        best_cost = cost2
        best_result = result2.copy()
    
    # Run local search on the best initial solution
    if best_result is not None:
        remaining = time_budget - (time.time() - global_start)
        if remaining > 5.0:
            ls_result, ls_cost = _local_search(best_result, demands, capacities, open_cost, assign_cost, time_limit=remaining - 5.0)
            if ls_cost < best_cost:
                best_cost = ls_cost
                best_result = ls_result
    
    # Try more local search iterations with different starting points if time permits
    elapsed = time.time() - global_start
    if elapsed < time_budget - 10.0:
        # Try random restarts
        np.random.seed(42)
        for restart in range(5):
            if time.time() - global_start > time_budget - 10.0:
                break
            # Perturb the best solution
            perturbed = best_result.copy()
            # Randomly reassign 5-10 customers
            n_perturb = np.random.randint(5, 11)
            indices = np.random.choice(n_cust, n_perturb, replace=False)
            for i in indices:
                # Assign to random feasible facility
                feasible = [f for f in range(n_fac) 
                           if np.sum(perturbed == f) * 0 + (np.sum(demands[perturbed == f]) + demands[i] <= capacities[f])]
                # Better: compute load properly
                load_pert = np.zeros(n_fac)
                for j, f in enumerate(perturbed):
                    load_pert[f] += demands[j]
                feasible = [f for f in range(n_fac) if load_pert[f] + demands[i] <= capacities[f]]
                if feasible:
                    perturbed[i] = np.random.choice(feasible)
            
            cost_pert = _total_cost(perturbed, demands, capacities, open_cost, assign_cost)
            if np.isinf(cost_pert):
                continue
            
            # Local search from perturbed
            remaining = time_budget - (time.time() - global_start) - 5.0
            if remaining > 2.0:
                ls_pert, ls_pert_cost = _local_search(perturbed, demands, capacities, open_cost, assign_cost, time_limit=remaining)
                if ls_pert_cost < best_cost:
                    best_cost = ls_pert_cost
                    best_result = ls_pert
    
    # Final safety check
    if best_result is None:
        # Fallback: assign all to facility 0 (may be infeasible, but better than nothing)
        best_result = np.zeros(n_cust, dtype=int)
    
    return best_result
# EVOLVE-BLOCK-END
