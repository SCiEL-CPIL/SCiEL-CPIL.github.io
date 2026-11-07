# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: MILP (HiGHS) with heuristic fallback and local-search polish.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # --- Step 1: Generate heuristic initial solutions ---
    fac_order = np.argsort(assign_cost, axis=1)
    
    def _greedy(order):
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                result[i] = 0
                load[0] += demands[i]
        return result
    
    def _compute_score(result, demands, capacities, open_cost, assign_cost):
        n_cust = len(demands)
        n_fac = len(capacities)
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities):
            return float('inf')
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            open_mask[f] = True
        score = np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), result])
        return score

    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Decreasing demand
    order1 = np.argsort(-demands)
    r1 = _greedy(order1)
    s1 = _compute_score(r1, demands, capacities, open_cost, assign_cost)
    if s1 < best_score:
        best_score = s1
        best_result = r1
    
    # Strategy 2: Natural order
    r2 = _greedy(np.arange(n_cust))
    s2 = _compute_score(r2, demands, capacities, open_cost, assign_cost)
    if s2 < best_score:
        best_score = s2
        best_result = r2
    
    # Strategy 3: Fewest feasible options first
    feasible_count = np.zeros(n_cust, dtype=np.int64)
    for i in range(n_cust):
        for f in range(n_fac):
            if demands[i] <= capacities[f]:
                feasible_count[i] += 1
    order3 = np.argsort(feasible_count)
    r3 = _greedy(order3)
    s3 = _compute_score(r3, demands, capacities, open_cost, assign_cost)
    if s3 < best_score:
        best_score = s3
        best_result = r3

    result = best_result.copy()
    
    # --- Step 2: Try MILP (HiGHS) ---
    # Variables: x[i, j] for i in 0..n_cust-1, j in 0..n_fac-1, then y[j] for j in 0..n_fac-1
    n_vars = n_cust * n_fac + n_fac
    
    # Objective
    c_obj = np.zeros(n_vars)
    for i in range(n_cust):
        for j in range(n_fac):
            c_obj[i * n_fac + j] = assign_cost[i, j]
    for j in range(n_fac):
        c_obj[n_cust * n_fac + j] = open_cost[j]
        
    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Constraints
    # 1. Assignment: sum_j x[i, j] = 1
    # 2. Capacity: sum_i demands[i] x[i, j] <= capacities[j]
    # 3. Linking: x[i, j] <= y[j]
    
    n_rows = n_cust + n_fac + n_cust * n_fac
    A = np.zeros((n_rows, n_vars))
    lb_cons = np.zeros(n_rows)
    ub_cons = np.zeros(n_rows)
    
    row_idx = 0
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
        lb_cons[row_idx] = 1.0
        ub_cons[row_idx] = 1.0
        row_idx += 1
        
    for j in range(n_fac):
        for i in range(n_cust):
            A[row_idx, i * n_fac + j] = demands[i]
        lb_cons[row_idx] = -np.inf
        ub_cons[row_idx] = capacities[j]
        row_idx += 1
        
    for i in range(n_cust):
        for j in range(n_fac):
            A[row_idx, i * n_fac + j] = 1.0
            A[row_idx, n_cust * n_fac + j] = -1.0
            lb_cons[row_idx] = -np.inf
            ub_cons[row_idx] = 0.0
            row_idx += 1
            
    constraints = LinearConstraint(A, lb_cons, ub_cons)
    bounds = Bounds(lb, ub)
    
    milp_solution = None
    try:
        if time.time() < deadline - 10:
            time_limit = max(1, deadline - time.time() - 5)
            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={'presolve': True, 'time_limit': time_limit})
            
            if res.status == 0:  # Optimal
                x_sol = res.x
                x_assign = x_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
                x_assign_rounded = np.round(x_assign).astype(int)
                
                valid = True
                load = np.zeros(n_fac, dtype=np.int64)
                result_milp = np.zeros(n_cust, dtype=np.int64)
                for i in range(n_cust):
                    assigned_facs = np.where(x_assign_rounded[i] == 1)[0]
                    if len(assigned_facs) != 1:
                        valid = False
                        break
                    f = assigned_facs[0]
                    result_milp[i] = f
                    load[f] += demands[i]
                
                if np.any(load > capacities):
                    valid = False
                    
                if valid:
                    milp_solution = result_milp
                    
            elif res.status in [1, 2, 3]:  # Feasible but not optimal, or limit reached
                if hasattr(res, 'x') and res.x is not None:
                    x_sol = res.x
                    x_assign = x_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
                    x_assign_rounded = np.round(x_assign).astype(int)
                    
                    valid = True
                    load = np.zeros(n_fac, dtype=np.int64)
                    result_milp = np.zeros(n_cust, dtype=np.int64)
                    for i in range(n_cust):
                        assigned_facs = np.where(x_assign_rounded[i] == 1)[0]
                        if len(assigned_facs) != 1:
                            # Try to find best facility based on fractional values
                            best_f = np.argmax(x_assign[i])
                            if x_assign[i, best_f] < 0.5:
                                valid = False
                                break
                            result_milp[i] = best_f
                            load[best_f] += demands[i]
                        else:
                            f = assigned_facs[0]
                            result_milp[i] = f
                            load[f] += demands[i]
                    
                    if valid and not np.any(load > capacities):
                        milp_solution = result_milp

    except Exception:
        pass
    
    if milp_solution is not None:
        s_milp = _compute_score(milp_solution, demands, capacities, open_cost, assign_cost)
        if s_milp < best_score:
            best_score = s_milp
            result = milp_solution.copy()
    
    # --- Step 3: Local Search Polish ---
    if time.time() < deadline - 5:
        result = _enhanced_local_search(result, demands, capacities, open_cost, assign_cost, deadline)
    
    # Final validity check
    result = _force_valid(result, demands, capacities, assign_cost)
    
    return result.astype(int)

def _force_valid(result, demands, capacities, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            result[i] = 0
        load[result[i]] += demands[i]
            
    if np.any(load > capacities):
        load = np.zeros(n_fac, dtype=np.int64)
        result = np.full(n_cust, -1, dtype=np.int64)
        order = np.argsort(-demands)
        fac_order = np.argsort(assign_cost, axis=1)
        
        for i in order:
            assigned = False
            for f in fac_order[i]:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                result[i] = 0
                load[0] += demands[i]
                
    return result

def _enhanced_local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    for i, f in enumerate(result):
        load[f] += demands[i]
        
    if np.any(load > capacities):
        for _ in range(1000):
            if time.time() > deadline:
                break
            over_facs = np.where(load > capacities)[0]
            if len(over_facs) == 0:
                break
            f_over = over_facs[0]
            best_move_cost = float('inf')
            best_i = -1
            best_f_new = -1
            
            custs_in_f = np.where(result == f_over)[0]
            for i in custs_in_f:
                for f_new in range(n_fac):
                    if f_new == f_over:
                        continue
                    if load[f_new] + demands[i] <= capacities[f_new]:
                        cost = assign_cost[i, f_new]
                        if cost < best_move_cost:
                            best_move_cost = cost
                            best_i = i
                            best_f_new = f_new
            
            if best_i != -1:
                load[f_over] -= demands[best_i]
                load[best_f_new] += demands[best_i]
                result[best_i] = best_f_new
            else:
                break
    
    current_obj = _compute_score(result, demands, capacities, open_cost, assign_cost)
    
    max_outer_iters = 20
    outer_iter = 0
    improved = True
    
    while improved and outer_iter < max_outer_iters and time.time() < deadline - 5:
        improved = False
        outer_iter += 1
        
        # --- Phase 1: Single customer moves ---
        cur_assign_cost = np.array([assign_cost[i, result[i]] for i in range(n_cust)])
        delta_base = assign_cost - cur_assign_cost[:, np.newaxis]
        
        cur_fac = result
        is_last_at_cur = (load[cur_fac] == demands)
        zero_load_fac = (load == 0)
        
        not_cur = np.ones((n_cust, n_fac), dtype=bool)
        not_cur[np.arange(n_cust), cur_fac] = False
        
        term1 = np.where(is_last_at_cur[:, np.newaxis], -open_cost[cur_fac][:, np.newaxis], 0.0)
        term1 = np.where(not_cur, term1, 0.0)
        
        term2 = np.where(zero_load_fac[np.newaxis, :], open_cost[np.newaxis, :], 0.0)
        term2 = np.where(not_cur, term2, 0.0)
        
        delta_open = term1 + term2
        total_delta = delta_base + delta_open
        
        total_delta[np.arange(n_cust), cur_fac] = np.inf
        
        infeasible = (load[np.newaxis, :] + demands[:, np.newaxis]) > capacities[np.newaxis, :]
        total_delta[infeasible] = np.inf
        
        best_f = np.argmin(total_delta, axis=1)
        best_delta = np.min(total_delta, axis=1)
        
        improving = best_delta < -1e-9
        if np.any(improving):
            improving_indices = np.where(improving)[0]
            improving_indices = improving_indices[np.argsort(best_delta[improving_indices])]
            
            modified_fac = np.zeros(n_fac, dtype=bool)
            modified_cust = np.zeros(n_cust, dtype=bool)
            
            for i in improving_indices:
                if time.time() > deadline - 5:
                    break
                if modified_cust[i]:
                    continue
                f_new = best_f[i]
                f_old = result[i]
                if modified_fac[f_new] or modified_fac[f_old]:
                    continue
                
                if load[f_new] + demands[i] > capacities[f_new]:
                    continue
                
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                modified_fac[f_old] = True
                modified_fac[f_new] = True
                modified_cust[i] = True
                improved = True
        
        # --- Phase 2: Pairwise swaps ---
        if outer_iter % 3 == 0 and time.time() < deadline - 10:
            np.random.seed(42 + outer_iter)
            sample_size = min(300, n_cust * (n_cust - 1) // 2)
            
            for _ in range(sample_size):
                if time.time() > deadline - 5:
                    break
                
                i1, i2 = np.random.choice(n_cust, size=2, replace=False)
                f1, f2 = result[i1], result[i2]
                
                if f1 == f2:
                    continue
                
                if load[f1] - demands[i1] + demands[i2] <= capacities[f1] and \
                   load[f2] - demands[i2] + demands[i1] <= capacities[f2]:
                    
                    old_cost = assign_cost[i1, f1] + assign_cost[i2, f2]
                    new_cost = assign_cost[i1, f2] + assign_cost[i2, f1]
                    
                    old_open1 = (load[f1] == demands[i1])
                    old_open2 = (load[f2] == demands[i2])
                    new_load_f1 = load[f1] - demands[i1] + demands[i2]
                    new_load_f2 = load[f2] - demands[i2] + demands[i1]
                    new_open1 = (new_load_f1 == 0)
                    new_open2 = (new_load_f2 == 0)
                    
                    delta_assign = new_cost - old_cost
                    delta_open = 0.0
                    if old_open1 and not new_open1:
                        delta_open += open_cost[f1]
                    elif not old_open1 and new_open1:
                        delta_open -= open_cost[f1]
                    if old_open2 and not new_open2:
                        delta_open += open_cost[f2]
                    elif not old_open2 and new_open2:
                        delta_open -= open_cost[f2]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        load[f1] = new_load_f1
                        load[f2] = new_load_f2
                        result[i1] = f2
                        result[i2] = f1
                        current_obj += total_delta
                        improved = True
                        
    return result

# EVOLVE-BLOCK-END
