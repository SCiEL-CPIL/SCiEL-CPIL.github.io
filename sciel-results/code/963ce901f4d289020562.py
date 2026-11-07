"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _evaluate_assignment(result, demands, capacities, open_cost, assign_cost):
    """Compute the objective value for a given assignment. Returns inf if invalid."""
    n_fac = len(capacities)
    if len(result) != len(demands):
        return float('inf')
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    open_facilities = np.unique(result)
    cost = np.sum(open_cost[open_facilities]) + np.sum(assign_cost[np.arange(len(demands)), result])
    return cost


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = []
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)
        if not feasible:
            # Fallback: pick facility with most remaining capacity
            remaining = capacities - load
            f = np.argmax(remaining)
            if remaining[f] >= demands[i]:
                load[f] += demands[i]
                result[i] = f
            else:
                # Should not happen if problem is feasible, but handle gracefully
                result[i] = 0
        else:
            f = feasible[0]
            load[f] += demands[i]
            result[i] = f
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=5000):
    """Local search: try reassigning customers to reduce cost."""
    n_cust = len(demands)
    n_fac = len(capacities)
    current = result.copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(current):
        load[f] += demands[i]
    
    # Compute current cost
    open_facilities = set(current.tolist())
    current_cost = np.sum(open_cost[list(open_facilities)]) + np.sum(assign_cost[np.arange(n_cust), current])
    
    best_cost = current_cost
    best_result = current.copy()
    
    for iteration in range(max_iters):
        improved = False
        for i in range(n_cust):
            current_fac = current[i]
            # Try all other facilities
            for f in range(n_fac):
                if f == current_fac:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    # Calculate cost change
                    # Remove customer i from current_fac
                    old_load_cf = load[current_fac]
                    load[current_fac] -= demands[i]
                    load[f] += demands[i]
                    
                    # Determine which facilities are open before and after
                    old_open = set(current.tolist())
                    old_open.add(current_fac)  # current_fac was open
                    # Check if current_fac becomes empty
                    if load[current_fac] == 0:
                        old_open.discard(current_fac)
                    
                    new_open = set(current.tolist())
                    new_open.add(f)
                    # Check if f was already open
                    # Actually, let's compute properly
                    new_assign = current.copy()
                    new_assign[i] = f
                    new_open_set = set(new_assign.tolist())
                    
                    # Cost calculation
                    old_open_set = set(current.tolist())
                    # current_fac is in old_open_set since it had customer i
                    # After removing i, if load[current_fac]==0, it's no longer open
                    if load[current_fac] == 0:
                        old_open_set.discard(current_fac)
                    
                    new_cost = np.sum(open_cost[list(new_open_set)]) + np.sum(assign_cost[np.arange(n_cust), new_assign])
                    
                    if new_cost < best_cost - 1e-9:
                        best_cost = new_cost
                        best_result = new_assign.copy()
                        # Update current state
                        current = new_assign.copy()
                        load = np.zeros(n_fac)
                        for j, ff in enumerate(current):
                            load[ff] += demands[j]
                        improved = True
                        break
                    else:
                        # Revert
                        load[f] -= demands[i]
                        load[current_fac] += demands[i]
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_result


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Initial greedy solution
    result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    # Local search improvement
    if time.time() < deadline:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=2000)
    
    return result
# EVOLVE-BLOCK-END
