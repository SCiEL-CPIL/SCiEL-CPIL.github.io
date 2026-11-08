"""Capacitated facility location: exact min-cost flow + greedy/local search fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linprog


def _solve_min_cost_flow(demands, capacities, open_cost, assign_cost, deadline):
    """
    Solve CFLP as a min-cost flow / LP problem.
    
    Formulation:
    - Binary variable x[i,f]: customer i assigned to facility f
    - Binary variable y[f]: facility f is open
    
    Constraints:
    - Each customer assigned to exactly one facility: sum_f x[i,f] = 1
    - Capacity: sum_i demands[i]*x[i,f] <= capacities[f]*y[f]
    - x[i,f] <= y[f] (can only assign if open)
    
    Objective: sum_f open_cost[f]*y[f] + sum_{i,f} assign_cost[i,f]*x[i,f]
    
    Since the constraint matrix is totally unimodular for this structure,
    the LP relaxation gives integer solutions.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Variables: x[i,f] for all i,f (n_cust*n_fac vars), then y[f] for all f (n_fac vars)
    # Total vars: n_cust*n_fac + n_fac
    n_vars = n_cust * n_fac + n_fac
    
    # Objective coefficients
    c = np.zeros(n_vars)
    for i in range(n_cust):
        for f in range(n_fac):
            c[i * n_fac + f] = assign_cost[i, f]
    for f in range(n_fac):
        c[n_cust * n_fac + f] = open_cost[f]
    
    # Equality constraints: sum_f x[i,f] = 1 for each customer i
    A_eq = np.zeros((n_cust, n_vars))
    for i in range(n_cust):
        for f in range(n_fac):
            A_eq[i, i * n_fac + f] = 1.0
    b_eq = np.ones(n_cust)
    
    # Inequality constraints:
    # 1) Capacity: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0 for each f
    # 2) x[i,f] - y[f] <= 0 for each i,f
    n_ineq = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    # Capacity constraints
    for f in range(n_fac):
        for i in range(n_cust):
            A_ub[f, i * n_fac + f] = demands[i]
        A_ub[f, n_cust * n_fac + f] = -capacities[f]
        b_ub[f] = 0.0
    
    # x[i,f] <= y[f]
    idx = n_fac
    for i in range(n_cust):
        for f in range(n_fac):
            A_ub[idx, i * n_fac + f] = 1.0
            A_ub[idx, n_cust * n_fac + f] = -1.0
            b_ub[idx] = 0.0
            idx += 1
    
    # Bounds: 0 <= x[i,f] <= 1, 0 <= y[f] <= 1
    bounds = [(0, 1)] * n_vars
    
    t_start = time.time()
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                      bounds=bounds, method='highs')
        
        if res.status == 0 and res.fun is not None:
            # Extract solution
            result = np.zeros(n_cust, dtype=int)
            for i in range(n_cust):
                best_f = 0
                best_val = -1
                for f in range(n_fac):
                    val = res.x[i * n_fac + f]
                    if val > best_val:
                        best_val = val
                        best_f = f
                result[i] = best_f
            
            # Verify validity
            load = np.zeros(n_fac)
            for i in range(n_cust):
                load[result[i]] += demands[i]
            
            valid = True
            for f in range(n_fac):
                if load[f] > capacities[f] + 1e-6:
                    valid = False
                    break
                if load[f] > 1e-6 and res.x[n_cust * n_fac + f] < 0.5:
                    # Facility has load but y[f] < 0.5 - should not happen with TU
                    valid = False
                    break
            
            if valid and all(0 <= result[i] < n_fac for i in range(n_cust)):
                return result, res.fun
            else:
                return None, None
        else:
            return None, None
    except Exception:
        return None, None


def _greedy_seed(demands, capacities, open_cost, assign_cost, strategy='demand_desc'):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)

    if strategy == 'demand_desc':
        total_cost = assign_cost.sum(axis=1)
        order = np.lexsort((total_cost, -demands))
    elif strategy == 'demand_asc':
        order = np.argsort(demands)
    elif strategy == 'random':
        order = np.random.permutation(n_cust)
    elif strategy == 'cheapest_first':
        min_costs = np.min(assign_cost, axis=1)
        order = np.argsort(min_costs)
    elif strategy == 'effective_cost':
        # Amortized open cost per unit demand
        amortized = open_cost / np.maximum(capacities, 1.0)
        eff_cost = assign_cost + np.outer(demands, amortized)
        total_eff = eff_cost.sum(axis=1)
        order = np.lexsort((total_eff, -demands))
    else:
        total_cost = assign_cost.sum(axis=1)
        order = np.lexsort((total_cost, -demands))

    fac_order = np.argsort(assign_cost, axis=1)

    for i in order:
        placed = False
        for f in fac_order[i]:
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f
    return result, load


def _objective(result, load, open_cost, assign_cost):
    used = load > 0
    open_part = open_cost[used].sum()
    assign_part = assign_cost[np.arange(len(result)), result].sum()
    return open_part + assign_part


def _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    fac_order = np.argsort(assign_cost, axis=1)

    cur_obj = _objective(result, load, open_cost, assign_cost)

    improved = True
    no_improve_pass = 0
    while improved and time.time() < deadline:
        improved = False
        
        # Single reassignment pass
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            f_old = result[i]
            if f_old < 0:
                continue
            best_gain = 0.0
            best_f = -1
            base_assign = assign_cost[i, f_old]
            for f_new in fac_order[i]:
                if f_new == f_old:
                    continue
                if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                    continue
                da = assign_cost[i, f_new] - base_assign
                do = 0.0
                if load[f_old] - demands[i] <= 1e-9:
                    do -= open_cost[f_old]
                if load[f_new] <= 1e-9:
                    do += open_cost[f_new]
                gain = -(da + do)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_f = f_new
            if best_f >= 0:
                load[f_old] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                cur_obj -= best_gain
                improved = True

        # Swap pass
        if time.time() < deadline:
            for i in range(n_cust):
                if time.time() >= deadline:
                    break
                fi = result[i]
                if fi < 0:
                    continue
                di = demands[i]
                for j in range(i + 1, n_cust):
                    if time.time() >= deadline:
                        break
                    fj = result[j]
                    if fj < 0 or fj == fi:
                        continue
                    dj = demands[j]
                    if load[fi] - di + dj > capacities[fi] + 1e-9:
                        continue
                    if load[fj] - dj + di > capacities[fj] + 1e-9:
                        continue
                    da = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                    do = 0.0
                    if load[fi] - di + dj <= 1e-9 and load[fi] > 1e-9:
                        do -= open_cost[fi]
                    if load[fj] - dj + di <= 1e-9 and load[fj] > 1e-9:
                        do -= open_cost[fj]
                    gain = -(da + do)
                    if gain > 1e-12:
                        load[fi] = load[fi] - di + dj
                        load[fj] = load[fj] - dj + di
                        result[i], result[j] = fj, fi
                        cur_obj -= gain
                        improved = True

        if not improved:
            no_improve_pass += 1
            if no_improve_pass >= 1:
                break
        else:
            no_improve_pass = 0

    return result, load, cur_obj


def _simulated_annealing(demands, capacities, open_cost, assign_cost, result, load, deadline):
    """Simulated annealing to escape local optima."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = load.copy()
    
    cur_obj = _objective(result, load, open_cost, assign_cost)
    best_obj = cur_obj
    best_result = result.copy()
    best_load = load.copy()
    
    T = 100.0
    cooling = 0.95
    min_T = 0.1
    
    t_start = time.time()
    while T > min_T and time.time() < deadline:
        # Pick a random customer
        i = random.randint(0, n_cust - 1)
        f_old = result[i]
        
        # Pick a random feasible facility
        feasible = []
        for f in range(n_fac):
            if f == f_old:
                continue
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                feasible.append(f)
        
        if not feasible:
            T *= cooling
            continue
        
        f_new = random.choice(feasible)
        
        # Compute delta
        da = assign_cost[i, f_new] - assign_cost[i, f_old]
        do = 0.0
        if load[f_old] - demands[i] <= 1e-9:
            do -= open_cost[f_old]
        if load[f_new] <= 1e-9:
            do += open_cost[f_new]
        delta = da + do
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            load[f_old] -= demands[i]
            load[f_new] += demands[i]
            result[i] = f_new
            cur_obj += delta
            
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_result = result.copy()
                best_load = load.copy()
        
        T *= cooling
    
    return best_result, best_load, best_obj


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Try exact min-cost flow first
    best_result = None
    best_cost = float('inf')
    
    if time.time() < deadline - 30.0:
        flow_result, flow_cost = _solve_min_cost_flow(demands, capacities, open_cost, assign_cost, deadline - 30.0)
        if flow_result is not None:
            best_result = flow_result
            best_cost = flow_cost
    
    # Fallback / improvement: greedy + local search
    if time.time() < deadline - 10.0:
        strategies = ['demand_desc', 'effective_cost', 'cheapest_first', 'random']
        for strategy in strategies:
            if time.time() > deadline - 15.0:
                break

            result, load = _greedy_seed(demands, capacities, open_cost, assign_cost, strategy)
            current_cost = _objective(result, load, open_cost, assign_cost)
            
            # Local search
            result, load, current_cost = _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline - 10.0)
            
            # Simulated annealing for further improvement
            if time.time() < deadline - 5.0:
                result, load, current_cost = _simulated_annealing(demands, capacities, open_cost, assign_cost, result, load, deadline - 5.0)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()

    if best_result is None:
        best_result, _ = _greedy_seed(demands, capacities, open_cost, assign_cost, 'demand_desc')

    # Final validation and repair
    result = best_result
    load_check = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        if f < 0 or f >= n_fac:
            f = int(np.argmin(load_check))
            result[i] = f
        load_check[f] += demands[i]

    # Repair capacity violations
    for f in range(n_fac):
        if load_check[f] > capacities[f] + 1e-9:
            excess = load_check[f] - capacities[f]
            custs_at_f = np.where(result == f)[0]
            custs_at_f = custs_at_f[np.argsort(-demands[custs_at_f])]
            for i in custs_at_f:
                if excess <= 1e-9:
                    break
                for f2 in np.argsort(assign_cost[i]):
                    if f2 == f:
                        continue
                    if load_check[f2] + demands[i] <= capacities[f2] + 1e-9:
                        load_check[f] -= demands[i]
                        load_check[f2] += demands[i]
                        result[i] = f2
                        excess -= demands[i]
                        break

    return result.astype(int)
# EVOLVE-BLOCK-END
