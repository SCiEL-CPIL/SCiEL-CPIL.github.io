# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _compute_total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    if np.any(load > capacities):
        return np.inf
    open_total = sum(open_cost[f] for f in set(result))
    assign_total = sum(assign_cost[i, result[i]] for i in range(len(result)))
    return open_total + assign_total

def _local_search_polish(result, demands, capacities, open_cost, assign_cost, deadline):
    """Improve solution via local search: try moving customers to other facilities."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    current_cost = _compute_total_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
    
    improved = True
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters and (deadline - time.time()) > 5.0:
        improved = False
        iter_count += 1
        
        # Try moving each customer to a better facility
        for i in range(n_cust):
            old_f = result[i]
            old_load = load[old_f]
            
            # Try all other facilities
            best_f = old_f
            best_delta = 0.0
            
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                # Check capacity
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Compute cost delta
                delta = assign_cost[i, new_f] - assign_cost[i, old_f]
                
                # Account for open cost changes
                # If old_f becomes empty, we save open_cost[old_f]
                # If new_f was empty, we pay open_cost[new_f]
                old_becomes_empty = (old_load == demands[i])
                new_was_empty = (load[new_f] == 0)
                
                if old_becomes_empty:
                    delta -= open_cost[old_f]
                if new_was_empty:
                    delta += open_cost[new_f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = new_f
            
            if best_f != old_f:
                # Make the move
                load[old_f] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                current_cost += best_delta
                improved = True
    
    return result

def _solve_milp(demands, capacities, open_cost, assign_cost, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    n_var = n_cust * n_fac + n_fac
    
    # Objective coefficients
    c = np.zeros(n_var)
    # x_{i,f} coefficients
    c[:n_cust * n_fac] = assign_cost.flatten()
    # y_f coefficients
    c[n_cust * n_fac:] = open_cost
    
    # Constraints
    # 1. Assignment constraints: sum_f x_{i,f} = 1 for each i
    A_eq = np.zeros((n_cust, n_var))
    for i in range(n_cust):
        A_eq[i, i*n_fac:(i+1)*n_fac] = 1
    b_eq = np.ones(n_cust)
    
    # 2. Capacity constraints: sum_i demands_i * x_{i,f} - capacities_f * y_f <= 0
    A_ub = np.zeros((n_fac, n_var))
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub[f, i*n_fac + f] = demands[i]
        A_ub[f, n_cust * n_fac + f] = -capacities[f]
    b_ub = np.zeros(n_fac)
    
    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
    ]
    
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary
    integrality = np.ones(n_var)
    
    time_limit = max(1.0, min(120.0, deadline - time.time()))
    
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.01}
        )
        if res.status in [0, 2, 3]:  # Optimal, time limit, or gap limit
            if res.x is not None:
                x_vals = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                # Find best facility for each customer
                result = np.argmax(x_vals, axis=1)
                # Verify solution validity just in case
                load = np.zeros(n_fac)
                for i in range(n_cust):
                    load[result[i]] += demands[i]
                if np.all(load <= capacities):
                    return result.astype(int)
    except Exception as e:
        pass
    return None

def _greedy_fallback(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1, dtype=int)
    order = np.argsort(-demands) # Largest first
    for i in order:
        # Sort facilities by (assign cost + marginal open cost)
        cand = np.argsort(assign_cost[i])
        for f in cand:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    # Fallback if any unassigned
    for i in range(n_cust):
        if result[i] < 0:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    return result.astype(int)

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    start = time.time()
    deadline = start + 135.0 # Safety margin (5s before 140s hard limit)
    
    result = _solve_milp(demands, capacities, open_cost, assign_cost, deadline)
    
    if result is None:
        result = _greedy_fallback(demands, capacities, open_cost, assign_cost)
    
    # Polish with local search if we have time
    remaining = deadline - time.time()
    if remaining > 10.0:
        result = _local_search_polish(result, demands, capacities, open_cost, assign_cost, deadline)
    
    return result
# EVOLVE-BLOCK-END
