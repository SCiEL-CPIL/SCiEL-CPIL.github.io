"""Capacitated facility location solver using LP relaxation + rounding + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _solve_lp_relaxation(demands, capacities, open_cost, assign_cost):
    """Solve LP relaxation of CFLP.
    Variables: x_{i,f} for i in [0,n), f in [0,m), and y_f for f in [0,m)
    Total variables: n*m + m
    Minimize: sum_f y_f * open_cost_f + sum_{i,f} x_{i,f} * assign_cost[i,f]
    Subject to:
      sum_f x_{i,f} = 1 for all i  (n constraints)
      sum_i x_{i,f} * demands[i] <= y_f * capacities[f] for all f  (m constraints)
      x_{i,f} >= 0, y_f >= 0
    Returns: (x_matrix, y_vector, lp_cost) or None if infeasible
    """
    n = len(demands)
    m = len(capacities)
    
    # Variable ordering: x_{0,0}, x_{0,1}, ..., x_{0,m-1}, x_{1,0}, ..., x_{n-1,m-1}, y_0, ..., y_{m-1}
    n_vars = n * m + m
    
    # Objective: c_j
    c = np.zeros(n_vars)
    for i in range(n):
        for f in range(m):
            c[i * m + f] = assign_cost[i, f]
    for f in range(m):
        c[n * m + f] = open_cost[f]
    
    # Equality constraints: sum_f x_{i,f} = 1 for each i
    A_eq = np.zeros((n, n_vars))
    b_eq = np.ones(n)
    for i in range(n):
        for f in range(m):
            A_eq[i, i * m + f] = 1.0
    
    # Inequality constraints: sum_i x_{i,f} * demands[i] - y_f * capacities[f] <= 0 for each f
    A_ub = np.zeros((m, n_vars))
    b_ub = np.zeros(m)
    for f in range(m):
        for i in range(n):
            A_ub[f, i * m + f] = demands[i]
        A_ub[f, n * m + f] = -capacities[f]
    
    # Bounds: all variables >= 0
    bounds = [(0, None)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    except Exception:
        return None
    
    if not res.success:
        return None
    
    x_flat = res.x[:n * m].reshape(n, m)
    y_vec = res.x[n * m:]
    lp_cost = res.fun
    
    return x_flat, y_vec, lp_cost


def _round_lp_solution(x_flat, y_vec, demands, capacities, assign_cost, open_cost):
    """Round LP solution to a feasible integer assignment."""
    n = len(demands)
    m = len(capacities)
    
    # Greedy rounding: for each customer, pick facility with highest x_{i,f}
    # Then fix capacity violations by reassigning
    result = np.zeros(n, dtype=int)
    for i in range(n):
        result[i] = np.argmax(x_flat[i])
    
    # Check and fix capacity violations
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    # Iteratively fix over-capacity facilities
    max_fix_iter = 100
    for _ in range(max_fix_iter):
        violated = [f for f in range(m) if load[f] > capacities[f] + 1e-9]
        if not violated:
            break
        # For the most violated facility, move a customer to another facility
        worst_f = max(violated, key=lambda f: load[f] - capacities[f])
        # Find customer at worst_f to move
        customers_at_worst = [i for i in range(n) if result[i] == worst_f]
        best_move = None
        best_move_cost = np.inf
        for i in customers_at_worst:
            for f in range(m):
                if f == worst_f:
                    continue
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    # Cost of moving i from worst_f to f
                    move_cost = assign_cost[i, f] - assign_cost[i, worst_f]
                    # Also consider open cost changes
                    if load[worst_f] == demands[i]:
                        move_cost -= open_cost[worst_f]
                    if load[f] == 0:
                        move_cost += open_cost[f]
                    if move_cost < best_move_cost:
                        best_move_cost = move_cost
                        best_move = (i, f)
        if best_move is None:
            # Force move to facility with most remaining capacity
            i = customers_at_worst[0]
            remaining = capacities - load
            f = np.argmax(remaining)
            best_move = (i, f)
        i, f = best_move
        load[result[i]] -= demands[i]
        load[f] += demands[i]
        result[i] = f
    
    return result


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            return float('inf')
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return float('inf')
    
    open_cost_total = 0.0
    assign_cost_total = 0.0
    for f in range(n_facilities):
        if load[f] > 0:
            open_cost_total += open_cost[f]
    
    for i in range(n_customers):
        assign_cost_total += assign_cost[i, result[i]]
    
    return open_cost_total + assign_cost_total


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000):
    """Local search: try moving single customers to other facilities."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = result.copy()
    best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)
    
    if not np.isfinite(best_cost):
        return best_result
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[best_result[i]] += demands[i]
    
    improved = True
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        for i in range(n_customers):
            current_f = best_result[i]
            current_load_f = load[current_f]
            
            best_alt_f = -1
            best_delta = 0
            
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if load[f] + demands[i] > capacities[f] + 1e-9:
                    continue
                
                delta = -assign_cost[i, current_f] + assign_cost[i, f]
                
                if current_load_f == demands[i]:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_alt_f = f
            
            if best_alt_f >= 0:
                load[current_f] -= demands[i]
                load[best_alt_f] += demands[i]
                best_result[i] = best_alt_f
                best_cost += best_delta
                improved = True
    
    return best_result


def _greedy_assign(demands, capacities, open_cost, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in np.argsort(-demands):
        order = np.argsort(assign_cost[i])
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            remaining = capacities - load
            f = np.argmax(remaining)
            load[f] += demands[i]
            result[i] = f
    return result


def assign(demands, capacities, open_cost, assign_cost):
    """Assign customers to facilities minimizing total cost."""
    start_time = time.time()
    deadline = start_time + 140
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Phase 1: LP relaxation + rounding
    result = None
    if time.time() < deadline - 60:
        lp_sol = _solve_lp_relaxation(demands, capacities, open_cost, assign_cost)
        if lp_sol is not None:
            x_flat, y_vec, lp_cost = lp_sol
            result = _round_lp_solution(x_flat, y_vec, demands, capacities, assign_cost, open_cost)
    
    if result is None:
        result = _greedy_assign(demands, capacities, open_cost, assign_cost)
    
    # Phase 2: Local search
    if time.time() < deadline - 20:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=10000)
    
    # Phase 3: Another local search pass
    if time.time() < deadline - 5:
        result = _local_search(result, demands, capacities, open_cost, assign_cost, max_iter=5000)
    
    # Ensure validity
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            result = _greedy_assign(demands, capacities, open_cost, assign_cost)
            break
        load[f] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            result = _greedy_assign(demands, capacities, open_cost, assign_cost)
            break
    
    return result.astype(int)
# EVOLVE-BLOCK-END
