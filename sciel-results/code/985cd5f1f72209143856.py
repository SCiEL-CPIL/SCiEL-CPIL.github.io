"""Capacitated facility location: assign 100 customers to 20 facilities."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    n_cust = len(result)
    total = float(np.sum(assign_cost[np.arange(n_cust), result]))
    used = np.unique(result)
    total += float(np.sum(open_cost[used]))
    return total

def _is_valid(result, demands, capacities):
    n_fac = len(capacities)
    if np.any(result < 0) or np.any(result >= n_fac):
        return False
    load = np.zeros(n_fac, dtype=np.float64)
    for i, f in enumerate(result):
        load[f] += demands[i]
    return bool(np.all(load <= capacities + 1e-9))

def _greedy_init(demands, capacities, open_cost, assign_cost, rng):
    """Greedy: largest demand first, cheapest feasible facility."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.float64)
    order = np.argsort(-demands, kind='stable')
    for i in order:
        d = demands[i]
        costs = assign_cost[i]
        feasible = [f for f in range(n_fac) if load[f] + d <= capacities[f] + 1e-9]
        if not feasible:
            f = int(np.argmax(capacities - load))
            load[f] += d
            result[i] = f
            continue
        fs = np.array(feasible)
        key = costs[fs] + rng.uniform(0, 1e-8, size=fs.shape)
        best = fs[np.argmin(key)]
        load[best] += d
        result[i] = best
    return result, load

def _local_search(result, load, demands, capacities, open_cost, assign_cost, deadline):
    """First-improvement single-move + swap local search."""
    n_cust = len(demands)
    n_fac = len(capacities)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in np.random.permutation(n_cust):
            if time.time() > deadline:
                break
            cf = result[i]
            di = demands[i]
            moved = False
            for f in range(n_fac):
                if f == cf:
                    continue
                if load[f] + di > capacities[f] + 1e-9:
                    continue
                assign_delta = assign_cost[i, f] - assign_cost[i, cf]
                open_delta = 0.0
                if load[cf] == di:
                    open_delta -= open_cost[cf]
                if load[f] == 0:
                    open_delta += open_cost[f]
                if assign_delta + open_delta < -1e-9:
                    load[cf] -= di
                    load[f] += di
                    result[i] = f
                    improved = True
                    moved = True
                    break
            if moved:
                continue
            same = [j for j in range(n_cust) if j != i and result[j] == cf]
            if not same:
                continue
            for j in same:
                dj = demands[j]
                new_load_cf = load[cf] - di + dj
                new_load_f = load[f] + di - dj
                if new_load_cf > capacities[cf] + 1e-9:
                    continue
                if new_load_f > capacities[f] + 1e-9:
                    continue
                if new_load_cf < -1e-9 or new_load_f < -1e-9:
                    continue
                assign_delta = (assign_cost[i, f] + assign_cost[j, cf]) - (assign_cost[i, cf] + assign_cost[j, f])
                open_delta = 0.0
                if load[cf] == di:
                    open_delta -= open_cost[cf]
                if load[f] == 0:
                    open_delta -= open_cost[f]
                if assign_delta + open_delta < -1e-9:
                    load[cf] = new_load_cf
                    load[f] = new_load_f
                    result[i] = f
                    result[j] = cf
                    improved = True
                    break
    return result, load

def _solve_mip(demands, capacities, open_cost, assign_cost, time_limit):
    """
    Solve CFLP as a MILP using scipy.optimize.milp.
    Variables: x[i, j] binary (customer i assigned to facility j), y[j] binary (facility j open)
    Minimize: sum_j open_cost[j]*y[j] + sum_{i,j} assign_cost[i,j]*x[i,j]
    Subject to:
      sum_j x[i,j] = 1 for all i
      sum_i demands[i]*x[i,j] <= capacities[j]*y[j] for all j
      x[i,j] <= y[j] for all i,j
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Number of variables: n_cust*n_fac for x + n_fac for y
    n_vars = n_cust * n_fac + n_fac
    
    # Objective: minimize
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c[n_cust * n_fac + j] = open_cost[j]
    
    # Constraints in sparse form
    # 1. Assignment constraints: sum_j x[i,j] = 1 for each i
    # 2. Capacity constraints: sum_i demands[i]*x[i,j] - capacities[j]*y[j] <= 0 for each j
    # 3. Linking constraints: x[i,j] - y[j] <= 0 for each i,j
    
    n_cons = n_cust + n_fac + n_cust * n_fac
    
    A = lil_matrix((n_cons, n_vars), dtype=np.float64)
    
    # 1. Assignment: for each i, sum_j x[i,j] = 1
    for i in range(n_cust):
        for j in range(n_fac):
            A[i, i * n_fac + j] = 1.0
    
    # 2. Capacity: for each j, sum_i demands[i]*x[i,j] - capacities[j]*y[j] <= 0
    for j in range(n_fac):
        for i in range(n_cust):
            A[n_cust + j, i * n_fac + j] = demands[i]
        A[n_cust + j, n_cust * n_fac + j] = -capacities[j]
    
    # 3. Linking: for each i,j, x[i,j] - y[j] <= 0
    for i in range(n_cust):
        for j in range(n_fac):
            row = n_cust + n_fac + i * n_fac + j
            A[row, i * n_fac + j] = 1.0
            A[row, n_cust * n_fac + j] = -1.0
    
    # Bounds: all variables in [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality: x are binary (0), y are binary (0)
    integrality = np.zeros(n_vars)
    integrality[:] = 1  # all binary
    
    # Linear constraints:
    # Assignment: equality (1)
    # Capacity: <= (ub=0)
    # Linking: <= (ub=0)
    cons_lb = np.zeros(n_cons)
    cons_ub = np.zeros(n_cons)
    for i in range(n_cust):
        cons_lb[i] = 1.0
        cons_ub[i] = 1.0
    # capacity and linking already have lb=0, ub=0
    
    constraints = LinearConstraint(A.tocsr(), cons_lb, cons_ub)
    bounds = Bounds(lb, ub)
    
    # Solve with time limit
    start_time = time.time()
    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                  options={"time_limit": time_limit, "mip_rel_gap": 0.01})
    
    if result.status != 0 and result.status != 1:
        return None
    
    x = result.x
    if x is None:
        return None
    
    # Extract assignment
    assignment = np.zeros(n_cust, dtype=int)
    for i in range(n_cust):
        # Find the facility with x[i,j] closest to 1
        x_row = x[i * n_fac: (i+1) * n_fac]
        assignment[i] = int(np.argmax(x_row))
    
    return assignment

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    n_cust = len(demands)
    n_fac = len(capacities)

    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_result = None
    best_cost = np.inf
    rng = np.random.default_rng(42)

    # First, try MIP solver with a time limit
    mip_time_limit = min(60.0, deadline - time.time() - 5.0)
    if mip_time_limit > 5.0:
        try:
            mip_solution = _solve_mip(demands, capacities, open_cost, assign_cost, mip_time_limit)
            if mip_solution is not None and _is_valid(mip_solution, demands, capacities):
                c = _compute_cost(mip_solution, demands, capacities, open_cost, assign_cost)
                best_cost = c
                best_result = mip_solution.copy()
        except Exception:
            pass

    # If MIP didn't give a solution or we still have time, do greedy + local search
    remaining = deadline - time.time()
    if remaining > 5.0:
        for attempt in range(10):
            if time.time() > deadline - 2.0:
                break
            result, load = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
            if not _is_valid(result, demands, capacities):
                continue
            sub_deadline = min(deadline - 2.0, time.time() + remaining / 10.0)
            result, load = _local_search(result, load, demands, capacities, open_cost, assign_cost, sub_deadline)
            if not _is_valid(result, demands, capacities):
                continue
            c = _compute_cost(result, demands, capacities, open_cost, assign_cost)
            if c < best_cost:
                best_cost = c
                best_result = result.copy()

    if best_result is None:
        # deterministic fallback greedy
        best_result, _ = _greedy_init(demands, capacities, open_cost, assign_cost, np.random.default_rng(0))

    return best_result
# EVOLVE-BLOCK-END
