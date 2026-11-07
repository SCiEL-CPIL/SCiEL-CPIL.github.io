"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _solve_lp(demands, capacities, open_cost, assign_cost):
    """
    Solve the LP relaxation of the capacitated facility location problem.
    Returns fractional assignment (n_customers, n_facilities) and open fractions.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables: x[i,f] for assignment, y[f] for open
    # Total vars = n_cust * n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac
    
    # Objective: minimize sum assign_cost[i,f] * x[i,f] + sum open_cost[f] * y[f]
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
        c[n_cust * n_fac + f] = open_cost[f]
    
    # Constraints:
    # 1. Each customer assigned to exactly one facility: sum_f x[i,f] = 1
    A_eq = np.zeros((n_cust, n_vars))
    b_eq = np.ones(n_cust)
    for i in range(n_cust):
        for f in range(n_fac):
            A_eq[i, i * n_fac + f] = 1.0
    
    # 2. Capacity constraints: sum_i demands[i] * x[i,f] <= capacities[f]
    # 3. x[i,f] <= y[f] for all i,f
    
    # Inequality constraints: A_ub @ x <= b_ub
    n_ineq = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    # Capacity constraints
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub[f, i * n_fac + f] = demands[i]
        b_ub[f] = capacities[f]
    
    # x[i,f] - y[f] <= 0
    for i in range(n_cust):
        for f in range(n_fac):
            idx = n_fac + i * n_fac + f
            A_ub[idx, i * n_fac + f] = 1.0
            A_ub[idx, n_cust * n_fac + f] = -1.0
            b_ub[idx] = 0.0
    
    # Bounds: 0 <= x[i,f] <= 1, 0 <= y[f] <= 1
    bounds = [(0, 1)] * (n_cust * n_fac) + [(0, 1)] * n_fac
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return None, None
    
    x = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
    y = res.x[n_cust * n_fac:]
    return x, y


def _round_lp(x, y, demands, capacities):
    """
    Round the LP solution to a feasible integer assignment.
    Uses a greedy approach prioritizing high assignment values, then repairs.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Sort customers by demand descending for harder constraints first
    order = np.argsort(-demands)
    
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    open_fac = np.zeros(n_fac, dtype=bool)
    
    # First pass: assign based on LP guidance
    # For each customer in order, pick the facility with highest x[i,f] that's feasible
    for i in order:
        # Get sorted facilities by LP value descending
        fac_order = np.argsort(-x[i])
        assigned = False
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                open_fac[f] = True
                assigned = True
                break
        if not assigned:
            # Fallback: cheapest feasible
            fac_order2 = np.argsort(assign_cost[i])
            for f in fac_order2:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    open_fac[f] = True
                    assigned = True
                    break
            if not assigned:
                # Should not happen if total capacity >= total demand
                result[i] = 0
                load[0] += demands[i]
                open_fac[0] = True
    
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=500):
    """
    Improve the assignment via local search: try moving customers to other facilities.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[result[i]] += demands[i]
    
    def total_cost():
        open_set = set(result)
        cost = sum(open_cost[f] for f in open_set)
        for i in range(n_cust):
            cost += assign_cost[i, result[i]]
        return cost
    
    best_cost = total_cost()
    
    for iteration in range(max_iter):
        improved = False
        # Try moving each customer to a better facility
        for i in range(n_cust):
            current_f = result[i]
            current_cost_i = assign_cost[i, current_f]
            
            # Find best alternative facility
            best_f = -1
            best_delta = 0
            
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    delta_assign = assign_cost[i, f] - current_cost_i
                    # Check if opening cost changes
                    current_open = current_f in set(result)
                    new_open = f in set(result)
                    
                    # If we move from current_f, it might close
                    # If we move to f, it might open
                    # Simplified: just consider assignment cost delta for now
                    if delta_assign < best_delta:
                        best_delta = delta_assign
                        best_f = f
            
            if best_f >= 0 and best_delta < -1e-9:
                # Perform the move
                load[current_f] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                improved = True
        
        if not improved:
            break
    
    return result


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Start with greedy baseline
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result_greedy[i] = f
                break
    
    def _cost(r):
        open_set = set(r)
        c = sum(open_cost[f] for f in open_set)
        for i in range(n_cust):
            c += assign_cost[i, r[i]]
        return c
    
    best_result = result_greedy.copy()
    best_cost = _cost(best_result)
    
    # Try LP relaxation
    try:
        x, y = _solve_lp(demands, capacities, open_cost, assign_cost)
        if x is not None:
            result_lp = _round_lp(x, y, demands, capacities)
            if np.all(result_lp >= 0):
                # Verify feasibility
                load_check = np.zeros(n_fac)
                feasible = True
                for i in range(n_cust):
                    load_check[result_lp[i]] += demands[i]
                if np.all(load_check <= capacities + 1e-9):
                    cost_lp = _cost(result_lp)
                    if cost_lp < best_cost:
                        best_result = result_lp.copy()
                        best_cost = cost_lp
    except Exception:
        pass
    
    # Local search on best result
    if time.time() < deadline:
        result_ls = _local_search(best_result, demands, capacities, open_cost, assign_cost, max_iter=200)
        cost_ls = _cost(result_ls)
        if cost_ls < best_cost:
            best_result = result_ls.copy()
            best_cost = cost_ls
    
    # Final verification
    load_final = np.zeros(n_fac)
    for i in range(n_cust):
        f = best_result[i]
        if f < 0 or f >= n_fac:
            # Fallback to greedy
            return result_greedy
        load_final[f] += demands[i]
    if np.any(load_final > capacities + 1e-9):
        return result_greedy
    
    return best_result
# EVOLVE-BLOCK-END
