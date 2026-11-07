"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _greedy_init(demands, capacities, assign_cost):
    """Greedy: largest demand first, cheapest feasible facility."""
    n = len(demands)
    result = np.zeros(n, dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    return result, load


def _solve_lp(demands, capacities, open_cost, assign_cost, deadline):
    """
    Solve CFLP as a linear program:
    Variables: x[i,f] in {0,1} (customer i assigned to facility f), y[f] in {0,1} (facility f open)
    Since the constraint matrix is totally unimodular for this bipartite structure with
    capacity constraints, the LP relaxation gives integer solutions.
    
    Minimize: sum_f open_cost[f]*y[f] + sum_{i,f} assign_cost[i,f]*x[i,f]
    Subject to:
      sum_f x[i,f] = 1 for all i (each customer assigned exactly once)
      sum_i demands[i]*x[i,f] <= capacities[f]*y[f] for all f
      x[i,f] >= 0, y[f] >= 0
    """
    n = len(demands)
    n_fac = len(capacities)
    
    # Variable indexing: x[i,f] for i in range(n), f in range(n_fac), then y[f] for f in range(n_fac)
    # Total variables: n * n_fac + n_fac
    n_vars = n * n_fac + n_fac
    
    def x_idx(i, f):
        return i * n_fac + f
    
    def y_idx(f):
        return n * n_fac + f
    
    # Objective: minimize sum open_cost[f]*y[f] + sum assign_cost[i,f]*x[i,f]
    c = np.zeros(n_vars)
    for f in range(n_fac):
        c[y_idx(f)] = open_cost[f]
    for i in range(n):
        for f in range(n_fac):
            c[x_idx(i, f)] = assign_cost[i, f]
    
    # Inequality constraints: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0 for all f
    A_ub = np.zeros((n_fac, n_vars))
    b_ub = np.zeros(n_fac)
    for f in range(n_fac):
        for i in range(n):
            A_ub[f, x_idx(i, f)] = demands[i]
        A_ub[f, y_idx(f)] = -capacities[f]
    
    # Equality constraints: sum_f x[i,f] = 1 for all i
    A_eq = np.zeros((n, n_vars))
    b_eq = np.ones(n)
    for i in range(n):
        for f in range(n_fac):
            A_eq[i, x_idx(i, f)] = 1.0
    
    bounds = [(0, 1)] * n_vars
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status != 0:
        return None
    
    x = res.x[:n * n_fac].reshape(n, n_fac)
    # Round x to nearest integer (should be close to 0 or 1)
    x_rounded = np.round(x).astype(int)
    
    # Ensure each customer is assigned to exactly one facility
    for i in range(n):
        if x_rounded[i].sum() != 1:
            # Pick the facility with smallest assign_cost among those with positive x
            f_best = np.argmin(assign_cost[i])
            x_rounded[i] = 0
            x_rounded[i, f_best] = 1
    
    result = x_rounded.argmax(axis=1)
    return result


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin from 150s
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    # Try LP first
    result = _solve_lp(demands, capacities, open_cost, assign_cost, deadline)
    
    if result is None:
        # Fallback to greedy
        result, _ = _greedy_init(demands, capacities, assign_cost)
    else:
        # Validate capacity constraints
        n_fac = len(capacities)
        load = np.zeros(n_fac, dtype=float)
        for i, f in enumerate(result):
            load[f] += demands[i]
        for f in range(n_fac):
            if load[f] > capacities[f] + 1e-9:
                # LP solution violates capacity; fallback to greedy
                result, _ = _greedy_init(demands, capacities, assign_cost)
                break
    
    return result.astype(int)
# EVOLVE-BLOCK-END
