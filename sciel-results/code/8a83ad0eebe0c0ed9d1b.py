# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix


def _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline):
    """Fast greedy + local search fallback."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = -np.ones(n_cust, dtype=int)
    load = np.zeros(n_fac)
    open_used = np.zeros(n_fac, dtype=bool)
    
    order = np.argsort(-demands)
    for i in order:
        eff_cost = assign_cost[i].copy()
        for f in range(n_fac):
            if not open_used[f]:
                eff_cost[f] += open_cost[f]
        for f in np.argsort(eff_cost):
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                if not open_used[f]:
                    open_used[f] = True
                break
    
    if -1 in result:
        return None
    
    # Quick local search
    best_result = result.copy()
    best_cost = _eval(best_result, demands, capacities, open_cost, assign_cost)
    if np.isinf(best_cost):
        return best_result
    
    load = np.zeros(n_fac)
    for i, f in enumerate(best_result):
        load[f] += demands[i]
    fac_cust = {f: [] for f in range(n_fac)}
    for i, f in enumerate(best_result):
        fac_cust[f].append(i)
    
    rng = np.random.RandomState(42)
    max_iters = 3000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        i = rng.randint(n_cust)
        cf = best_result[i]
        cd = demands[i]
        is_last = len(fac_cust[cf]) == 1
        best_new = best_cost
        best_nf = -1
        for f in range(n_fac):
            if f == cf:
                continue
            if load[f] + cd > capacities[f] + 1e-9:
                continue
            da = assign_cost[i, f] - assign_cost[i, cf]
            do = 0.0
            if is_last:
                do -= open_cost[cf]
            if len(fac_cust[f]) == 0:
                do += open_cost[f]
            nc = best_cost + da + do
            if nc < best_new - 1e-9:
                best_new = nc
                best_nf = f
        if best_nf >= 0:
            load[cf] -= cd
            fac_cust[cf].remove(i)
            load[best_nf] += cd
            fac_cust[best_nf].append(i)
            best_result[i] = best_nf
            best_cost = best_new
    
    return best_result


def _eval(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    open_total = np.sum(open_cost[load > 0])
    assign_total = np.sum(assign_cost[np.arange(len(demands)), result])
    return open_total + assign_total


def _solve_milp(demands, capacities, open_cost, assign_cost, deadline):
    """
    Solve CFLP as MILP:
    Variables:
      y_f (binary) for f in 0..n_fac-1 : facility f is open
      x_{i,f} (binary) for i in 0..n_cust-1, f in 0..n_fac-1 : customer i assigned to f
    
    Minimize: sum_f open_cost_f * y_f + sum_{i,f} assign_cost_{i,f} * x_{i,f}
    Subject to:
      sum_f x_{i,f} = 1  for all i          (each customer assigned exactly once)
      sum_i demands_i * x_{i,f} <= capacity_f * y_f  for all f  (capacity + linking)
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variable ordering: x_0_0, x_0_1, ..., x_0_{F-1}, x_1_0, ..., x_{C-1}_{F-1}, y_0, ..., y_{F-1}
    # Total variables: n_cust * n_fac + n_fac
    n_x = n_cust * n_fac
    n_y = n_fac
    n_vars = n_x + n_y
    
    # Objective
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_x + f] = open_cost[f]
    
    # Bounds: all binary
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints
    # 1. Assignment: sum_f x_{i,f} = 1 for each i
    #    n_cust constraints, each row has n_fac entries (the x variables for that customer)
    A_rows = []
    A_cols = []
    A_vals = []
    
    for i in range(n_cust):
        for f in range(n_fac):
            A_rows.append(i)
            A_cols.append(i * n_fac + f)
            A_vals.append(1.0)
    
    # 2. Capacity linking: sum_i demands_i * x_{i,f} <= capacity_f * y_f  =>  sum_i demands_i * x_{i,f} - capacity_f * y_f <= 0
    #    n_fac constraints
    for f in range(n_fac):
        row_idx = n_cust + f
        for i in range(n_cust):
            A_rows.append(row_idx)
            A_cols.append(i * n_fac + f)
            A_vals.append(float(demands[i]))
        A_rows.append(row_idx)
        A_cols.append(n_x + f)
        A_vals.append(-float(capacities[f]))
    
    A = csr_matrix((A_vals, (A_rows, A_cols)), shape=(n_cust + n_fac, n_vars))
    
    # Lower bounds for constraints
    clb = np.concatenate([np.ones(n_cust), np.zeros(n_fac)])
    # Upper bounds: assignment constraints are equality (1,1), capacity constraints are <= (0, inf)
    cub = np.concatenate([np.ones(n_cust), np.zeros(n_fac)])
    
    constraints = LinearConstraint(A, clb, cub)
    bounds = Bounds(lb, ub)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n_vars)
    
    # Try to solve
    try:
        sol = milp(c, constraints=constraints, bounds=bounds, integrality=integrality,
                   options={"time_limit": max(1.0, deadline - time.time())})
        
        if sol.status == 0:  # Optimal
            x = sol.x[:n_x].reshape(n_cust, n_fac)
            result = np.argmax(x, axis=1).astype(int)
            # Verify
            cost = _eval(result, demands, capacities, open_cost, assign_cost)
            if not np.isinf(cost):
                return result, cost
        elif sol.status == 2:  # Feasible (incumbent found, not necessarily optimal)
            x = sol.x[:n_x].reshape(n_cust, n_fac)
            result = np.argmax(x, axis=1).astype(int)
            cost = _eval(result, demands, capacities, open_cost, assign_cost)
            if not np.isinf(cost):
                return result, cost
        # status 1: infeasible, 3: no feasible solution, 4: numerical issues, 5: time limit
    except Exception:
        pass
    
    return None, float('inf')


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to one facility using MILP (HiGHS) with heuristic fallback.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # First, get a quick feasible solution via heuristic (gives us a safety net)
    heuristic_result = _heuristic_fallback(demands, capacities, open_cost, assign_cost, deadline)
    heuristic_cost = float('inf')
    if heuristic_result is not None:
        heuristic_cost = _eval(heuristic_result, demands, capacities, open_cost, assign_cost)
    
    # Now try MILP
    milp_result, milp_cost = _solve_milp(demands, capacities, open_cost, assign_cost, deadline)
    
    if milp_result is not None and milp_cost < heuristic_cost:
        return milp_result
    elif heuristic_result is not None:
        return heuristic_result
    else:
        # Last resort: assign all to facility 0
        return np.zeros(n_cust, dtype=int)
# EVOLVE-BLOCK-END
