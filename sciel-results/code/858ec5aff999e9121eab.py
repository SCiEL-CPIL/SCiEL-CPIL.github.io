# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacities.
    Minimizes open_cost (for open facilities) + assign_cost.
    Strategy: LP Relaxation + Rounding + Repair + Local Search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin

    # If n_cust or n_fac is very small, fall back to brute force or simple greedy
    # But 100x20 is standard.

    # --- Step 1: Solve LP Relaxation ---
    # Variables:
    # x[i, j] : 1 if customer i is assigned to facility j (0..1 in LP)
    # y[j]    : 1 if facility j is open (0..1 in LP)
    # Total variables: n_cust * n_fac + n_fac
    
    # Objective: Minimize sum(open_cost[j] * y[j]) + sum(assign_cost[i,j] * x[i,j])
    # c vector for linprog
    c = np.zeros(n_cust * n_fac + n_fac)
    # x variables come first
    c[:n_cust * n_fac] = assign_cost.flatten()
    # y variables come last
    c[n_cust * n_fac:] = open_cost

    # Inequality Constraints: A_ub @ x <= b_ub
    # 1. Capacity constraints: sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    #    For each facility j:
    #    sum_i demands[i] * x[i,j] <= capacities[j] * y[j]
    #    sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
    
    # 2. Linking constraints: x[i,j] <= y[j]
    #    x[i,j] - y[j] <= 0
    
    # We can combine these into A_ub.
    # Number of rows in A_ub:
    # n_fac (capacity) + n_cust * n_fac (linking)
    n_rows = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_rows, n_cust * n_fac + n_fac))
    b_ub = np.zeros(n_rows)
    
    # Fill Capacity Constraints (first n_fac rows)
    for j in range(n_fac):
        # x[i,j] coefficients
        col_start = j
        col_step = n_fac
        for i in range(n_cust):
            idx = i * n_fac + j
            A_ub[j, idx] = demands[i]
        # y[j] coefficient
        A_ub[j, n_cust * n_fac + j] = -capacities[j]
        b_ub[j] = 0.0

    # Fill Linking Constraints (next n_cust * n_fac rows)
    # x[i,j] - y[j] <= 0
    row_idx = n_fac
    for i in range(n_cust):
        for j in range(n_fac):
            idx_x = i * n_fac + j
            idx_y = n_cust * n_fac + j
            A_ub[row_idx, idx_x] = 1.0
            A_ub[row_idx, idx_y] = -1.0
            b_ub[row_idx] = 0.0
            row_idx += 1

    # Equality Constraints: A_eq @ x = b_eq
    # sum_j x[i,j] = 1 for each customer i
    n_eq_rows = n_cust
    A_eq = np.zeros((n_eq_rows, n_cust * n_fac + n_fac))
    b_eq = np.ones(n_eq_rows)
    
    for i in range(n_cust):
        for j in range(n_fac):
            idx = i * n_fac + j
            A_eq[i, idx] = 1.0

    # Bounds
    # x[i,j] in [0, 1]
    # y[j] in [0, 1]
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)

    # Solve LP
    try:
        # Use HiGHS if available, else default
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to greedy if LP fails
            raise RuntimeError("LP failed")
        
        x_lp = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
        # y_lp = res.x[n_cust * n_fac:]
        
    except Exception:
        # Fallback to greedy if LP fails or times out
        x_lp = None

    if x_lp is not None:
        # --- Step 2: Rounding ---
        # For each customer, pick the facility with max x_lp[i,j]
        result = np.argmax(x_lp, axis=1).astype(np.int64)
        
        # --- Step 3: Repair Capacity Violations ---
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(result):
            load[f] += demands[i]
        
        # If any capacity is violated, we need to move customers.
        # We'll use a local search approach to fix violations and improve cost.
        # This is effectively the same local search as before, but starting from a better point.
        
        # Check if we are within time limit
        if time.time() < deadline - 10:
            result = _local_search_repair(result, demands, capacities, open_cost, assign_cost, deadline)
        else:
            # If out of time, just return the rounded solution even if invalid? 
            # No, invalid scores 0. We must ensure validity.
            # If LP rounding created violations, we must fix them.
            result = _force_valid(result, demands, capacities, assign_cost)
    else:
        # Fallback: Greedy + Local Search (same as original but maybe faster start)
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
                    return None
            return result
        
        order1 = np.argsort(-demands)
        r = _greedy(order1)
        if r is None:
            r = np.array([i % n_fac for i in range(n_cust)], dtype=np.int64)
            
        if time.time() < deadline - 10:
            r = _local_search_repair(r, demands, capacities, open_cost, assign_cost, deadline)
        else:
            r = _force_valid(r, demands, capacities, assign_cost)
        result = r

    return result.astype(int)

def _force_valid(result, demands, capacities, assign_cost):
    """
    Emergency repair: if result is invalid, make it valid by moving customers
    to any facility with space, prioritizing lowest assignment cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    
    # First pass: check validity
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            result[i] = 0
        load[result[i]] += demands[i]
            
    # If any capacity exceeded, we need to redistribute.
    # Simple strategy: sort customers by demand descending, reassign all.
    # This is a bit heavy but ensures validity.
    if np.any(load > capacities):
        # Reset and re-greedy
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
                # Find any facility with space
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        assigned = True
                        break
            if not assigned:
                # Should not happen if total capacity >= total demand
                result[i] = 0
                load[0] += demands[i]
                
    return result

def _local_search_repair(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Local search that ensures validity and minimizes cost.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = np.zeros(n_fac, dtype=np.int64)
    for i, f in enumerate(result):
        load[f] += demands[i]
        
    # Ensure valid start
    if np.any(load > capacities):
        # Fix violations first
        # Identify over-capacity facilities
        # Move customers from over to under
        # Simple loop: while any violation exists
        max_iters_fix = 1000
        for _ in range(max_iters_fix):
            if time.time() > deadline:
                break
            over_facs = np.where(load > capacities)[0]
            if len(over_facs) == 0:
                break
            # Pick the worst over-capacity facility
            # Actually, just pick the first one
            f_over = over_facs[0]
            # Find a customer in f_over to move
            # Pick the one with smallest "loss" or just the first one?
            # Better: pick the customer in f_over who can move to an under-capacity facility with min cost increase.
            moved = False
            best_move_cost = float('inf')
            best_i = -1
            best_f_new = -1
            
            # Get customers assigned to f_over
            custs_in_f = np.where(result == f_over)[0]
            
            for i in custs_in_f:
                # Try moving i to any facility f_new where load[f_new] + demands[i] <= capacities[f_new]
                for f_new in range(n_fac):
                    if f_new == f_over:
                        continue
                    if load[f_new] + demands[i] <= capacities[f_new]:
                        # Calculate cost change
                        # This is a rough estimate, we will refine in main LS loop
                        # For now, just pick the one with lowest assign_cost[i, f_new]
                        cost = assign_cost[i, f_new]
                        if cost < best_move_cost:
                            best_move_cost = cost
                            best_i = i
                            best_f_new = f_new
            
            if best_i != -1:
                # Apply move
                load[f_over] -= demands[best_i]
                load[best_f_new] += demands[best_i]
                result[best_i] = best_f_new
                moved = True
            else:
                # If no move found, break (should not happen if total cap ok)
                break
                
    # Now main local search for cost improvement
    # This is similar to the original _local_search but optimized
    
    def _compute_obj(res):
        load = np.zeros(n_fac, dtype=np.int64)
        for i, f in enumerate(res):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
        if np.any(load > capacities):
            return float('inf')
        open_mask = np.zeros(n_fac, dtype=bool)
        for f in res:
            open_mask[f] = True
        return np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_cust), res])

    current_obj = _compute_obj(result)
    
    max_iters = 200
    iters = 0
    improved = True
    while improved and iters < max_iters and time.time() < deadline - 5:
        improved = False
        iters += 1
        
        # Vectorized move evaluation
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
        if not np.any(improving):
            break
        
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
            
    return result

# EVOLVE-BLOCK-END
