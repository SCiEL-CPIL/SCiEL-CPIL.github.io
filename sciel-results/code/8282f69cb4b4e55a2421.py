"""Facility location solver using MILP via HiGHS with greedy fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result


def _milp_assign(demands, capacities, open_cost, assign_cost, deadline):
    """
    Solve the fixed-charge facility location problem using LP relaxation
    and rounding, or directly via HiGHS if available.
    
    We use scipy.optimize.linprog for LP relaxation, then round.
    """
    n_c = len(demands)
    n_f = len(capacities)
    
    # Variables: x_{i,j} for assignment (n_c * n_f), y_j for open (n_f)
    # Total vars: n_c * n_f + n_f
    
    # Objective: minimize sum_{i,j} assign_cost[i,j] * x_{i,j} + sum_j open_cost[j] * y_j
    c = np.concatenate([assign_cost.flatten(), open_cost])
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_j x_{i,j} = 1 for all i
    # 2. Capacity: sum_i demands[i] * x_{i,j} <= capacities[j] * y_j for all j
    #    => sum_i demands[i] * x_{i,j} - capacities[j] * y_j <= 0
    
    # Equality constraints for customer assignment
    A_eq = np.zeros((n_c, n_c * n_f + n_f))
    b_eq = np.ones(n_c)
    for i in range(n_c):
        for j in range(n_f):
            A_eq[i, i * n_f + j] = 1.0
    
    # Inequality constraints for capacity
    A_ub = np.zeros((n_f, n_c * n_f + n_f))
    b_ub = np.zeros(n_f)
    for j in range(n_f):
        for i in range(n_c):
            A_ub[j, i * n_f + j] = demands[i]
        A_ub[j, n_c * n_f + j] = -capacities[j]
    
    # Bounds: 0 <= x <= 1, 0 <= y <= 1
    bounds = [(0, 1)] * (n_c * n_f + n_f)
    
    t_start = time.time()
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                      bounds=bounds, method='highs')
    except Exception:
        return None
    
    if res.status != 0:
        return None
    
    elapsed = time.time() - t_start
    if elapsed > deadline:
        return None
    
    # Extract assignment from LP solution
    x = res.x[:n_c * n_f].reshape(n_c, n_f)
    y = res.x[n_c * n_f:]
    
    # Round: for each customer, assign to the facility with highest x value
    # But we need to ensure capacity constraints are met
    # Use a greedy rounding approach: sort customers by "confidence" and assign
    
    # First, try direct rounding
    result = np.zeros(n_c, dtype=int)
    load = np.zeros(n_f)
    
    # Sort customers by demand descending for safety
    order = np.argsort(-demands)
    
    assigned = np.zeros(n_c, dtype=bool)
    for i in order:
        # Get facilities sorted by x value descending
        facs = np.argsort(-x[i])
        placed = False
        for f in facs:
            if load[f] + demands[i] <= capacities[f]:
                result[i] = f
                load[f] += demands[i]
                assigned[i] = True
                placed = True
                break
        if not placed:
            # Fallback: cheapest feasible
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    assigned[i] = True
                    placed = True
                    break
    
    # Verify all assigned
    if not np.all(assigned):
        return None
    
    # Verify capacity
    for f in range(n_f):
        total = sum(demands[i] for i in range(n_c) if result[i] == f)
        if total > capacities[f] + 1e-6:
            return None
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_c = len(demands)
    n_f = len(capacities)
    
    # Get a valid greedy solution first
    greedy_result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    greedy_score = sum(open_cost[f] for f in range(n_f) if np.any(greedy_result == f)) + \
                   sum(assign_cost[i, greedy_result[i]] for i in range(n_c))
    
    # Try MILP with time limit
    deadline = 100.0  # Leave margin for 120s total
    t_start = time.time()
    
    milp_result = _milp_assign(demands, capacities, open_cost, assign_cost, deadline)
    
    if milp_result is not None:
        # Verify validity
        load = np.zeros(n_f)
        for i in range(n_c):
            load[milp_result[i]] += demands[i]
        if np.all(load <= capacities + 1e-6) and np.all(milp_result >= 0) and np.all(milp_result < n_f):
            milp_score = sum(open_cost[f] for f in range(n_f) if np.any(milp_result == f)) + \
                         sum(assign_cost[i, milp_result[i]] for i in range(n_c))
            if milp_score <= greedy_score:
                return milp_result.astype(int)
    
    # Check time
    if time.time() - t_start > 110:
        return greedy_result
    
    # Try local search improvement on the best solution
    best = greedy_result.copy()
    best_score = greedy_score
    
    # Simple local search: try moving each customer to a different facility
    t_ls_start = time.time()
    improved = True
    while improved and (time.time() - t_ls_start) < 100:
        improved = False
        load = np.zeros(n_f)
        for i in range(n_c):
            load[best[i]] += demands[i]
        
        for i in range(n_c):
            current_f = best[i]
            # Try all other facilities
            for f in range(n_f):
                if f == current_f:
                    continue
                # Check capacity
                if load[f] + demands[i] <= capacities[f]:
                    # Calculate score change
                    old_cost = assign_cost[i, current_f]
                    new_cost = assign_cost[i, f]
                    
                    # Open cost change
                    old_open = 0
                    if np.sum(best == current_f) == 1:
                        old_open = open_cost[current_f]
                    new_open = 0
                    if np.sum(best == f) == 0:
                        new_open = open_cost[f]
                    
                    delta = (new_cost - old_cost) + (new_open - old_open)
                    if delta < -1e-9:
                        # Apply move
                        load[current_f] -= demands[i]
                        load[f] += demands[i]
                        best[i] = f
                        best_score += delta
                        improved = True
                        break
    
    return best.astype(int)
# EVOLVE-BLOCK-END
