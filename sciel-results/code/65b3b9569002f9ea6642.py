"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    # Check validity
    if np.any(load > capacities + 1e-6):
        return np.inf
    
    # Opening costs
    open_facs = np.unique(result)
    open_cost_sum = np.sum(open_cost[open_facs])
    
    # Assignment costs
    assign_cost_sum = np.sum(assign_cost[np.arange(len(result)), result])
    
    return open_cost_sum + assign_cost_sum

def _is_valid(result, demands, capacities):
    """Check if assignment is valid (capacity constraints satisfied)."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return False
        load[f] += demands[i]
    return np.all(load <= capacities + 1e-6)

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """
    Improved greedy initialization with multiple strategies, returns best.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    best_result = None
    best_cost = np.inf
    
    # Strategy 1: Sort by demand descending
    order1 = np.argsort(-demands)
    # Strategy 2: Sort by minimum assignment cost ascending
    min_ac = np.min(assign_cost, axis=1)
    order2 = np.argsort(min_ac)
    # Strategy 3: Sort by demand / min_assign_cost ratio
    ratios = demands / (min_ac + 1e-9)
    order3 = np.argsort(ratios)
    
    for order in [order1, order2, order3]:
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        valid = True
        
        for i in order:
            best_f = -1
            best_cost_f = np.inf
            
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    if load[f] == 0:
                        # Estimate how many customers will fit
                        est_customers = max(1, int(capacities[f] / max(1, demands[i])))
                        cost = assign_cost[i, f] + open_cost[f] / est_customers
                    else:
                        cost = assign_cost[i, f]
                    
                    if cost < best_cost_f:
                        best_cost_f = cost
                        best_f = f
            
            if best_f >= 0:
                result[i] = best_f
                load[best_f] += demands[i]
            else:
                # Fallback: assign to facility with most remaining capacity
                rem = capacities - load
                best_f = np.argmax(rem)
                result[i] = best_f
                load[best_f] += demands[i]
        
        if _is_valid(result, demands, capacities):
            cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
    
    return best_result if best_result is not None else np.zeros(n_cust, dtype=int)

def _local_search_move(result, demands, capacities, open_cost, assign_cost, max_iters=50000):
    """
    Fast local search: try moving individual customers to better facilities.
    Uses vectorized cost delta computation.
    """
    n_cust = len(result)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    # Count customers per facility for quick empty-check
    fac_counts = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        fac_counts[result[i]] += 1
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for i in range(n_cust):
            old_f = result[i]
            
            # Find best facility to move to
            best_f = -1
            best_delta = -1e-9
            
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f] + 1e-6:
                    continue
                
                # Assignment cost delta
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                
                # Opening cost delta
                delta_open = 0.0
                # If old facility becomes empty
                if fac_counts[old_f] == 1:
                    delta_open -= open_cost[old_f]
                # If new facility was empty
                if fac_counts[new_f] == 0:
                    delta_open += open_cost[new_f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f = new_f
            
            if best_f >= 0:
                # Apply the move
                result[i] = best_f
                fac_counts[old_f] -= 1
                fac_counts[best_f] += 1
                load[old_f] -= demands[i]
                load[best_f] += demands[i]
                current_cost += best_delta
                improved = True
    
    return result

def _local_search_swap(result, demands, capacities, open_cost, assign_cost, max_iters=20000):
    """
    Local search: try swapping two customers between facilities.
    Optimized with early termination.
    """
    n_cust = len(result)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    fac_counts = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        fac_counts[result[i]] += 1
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Group customers by facility
        fac_customers = [[] for _ in range(n_fac)]
        for i in range(n_cust):
            fac_customers[result[i]].append(i)
        
        found = False
        for f1 in range(n_fac):
            if len(fac_customers[f1]) < 1 or found:
                continue
            for f2 in range(f1 + 1, n_fac):
                if len(fac_customers[f2]) < 1 or found:
                    continue
                
                for i in fac_customers[f1]:
                    if found:
                        break
                    for j in fac_customers[f2]:
                        if i == j:
                            continue
                        
                        # Check capacity after swap
                        if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-6:
                            continue
                        if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-6:
                            continue
                        
                        # Compute cost delta
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        # Opening cost change
                        delta_open = 0.0
                        new_count_f1 = fac_counts[f1]
                        new_count_f2 = fac_counts[f2]
                        # After swap, counts don't change (one leaves, one enters each)
                        # So no opening cost change unless a facility had exactly 1 customer
                        # and that customer is being swapped out (but another comes in, so count stays same)
                        # Actually counts stay the same, so no delta_open
                        
                        total_delta = delta_assign
                        
                        if total_delta < -1e-9:
                            # Accept the swap
                            result[i] = f2
                            result[j] = f1
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            current_cost += total_delta
                            improved = True
                            found = True
                            break
    
    return result

def _local_search_2opt(result, demands, capacities, open_cost, assign_cost, max_iters=10000):
    """
    2-opt style: move a customer from one facility to another, 
    possibly closing an empty facility.
    """
    n_cust = len(result)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    fac_counts = np.zeros(n_fac, dtype=int)
    for i in range(n_cust):
        fac_counts[result[i]] += 1
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for i in range(n_cust):
            old_f = result[i]
            
            best_f = -1
            best_delta = -1e-9
            
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f] + 1e-6:
                    continue
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                
                delta_open = 0.0
                if fac_counts[old_f] == 1:
                    delta_open -= open_cost[old_f]
                if fac_counts[new_f] == 0:
                    delta_open += open_cost[new_f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_f = new_f
            
            if best_f >= 0:
                result[i] = best_f
                fac_counts[old_f] -= 1
                fac_counts[best_f] += 1
                load[old_f] -= demands[i]
                load[best_f] += demands[i]
                current_cost += best_delta
                improved = True
    
    return result

def _refine_with_local_search(result, demands, capacities, open_cost, assign_cost, time_limit=60.0):
    """
    Apply local search moves and swaps to refine the solution.
    """
    start_time = time.time()
    
    # First pass: individual moves (fast convergence)
    result = _local_search_move(result, demands, capacities, open_cost, assign_cost, max_iters=30000)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.3:
        return result
    
    # Second pass: swaps
    result = _local_search_swap(result, demands, capacities, open_cost, assign_cost, max_iters=15000)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.5:
        return result
    
    # Third pass: more moves after swaps
    result = _local_search_move(result, demands, capacities, open_cost, assign_cost, max_iters=20000)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.75:
        return result
    
    # Fourth pass: final polish with swaps
    result = _local_search_swap(result, demands, capacities, open_cost, assign_cost, max_iters=10000)
    
    return result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Start with greedy initialization (multiple strategies)
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    best_result = result.copy()
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    
    # 2. Refine with local search first (fast, good quality)
    elapsed = time.time() - start_time
    ls_time = min(40.0, deadline - start_time - 50.0)  # Leave 50s for MIP + buffer
    
    if ls_time > 5.0:
        try:
            refined = _refine_with_local_search(result, demands, capacities, open_cost, assign_cost, 
                                                 time_limit=ls_time)
            if _is_valid(refined, demands, capacities):
                cost_refined = _compute_cost(refined, demands, capacities, open_cost, assign_cost)
                if cost_refined < best_cost:
                    best_result = refined
                    best_cost = cost_refined
                    result = refined
        except Exception:
            pass
    
    # 3. Try MIP with time limit, using local search result as warm start
    try:
        elapsed = time.time() - start_time
        mip_time_limit = min(50.0, deadline - start_time - 15.0)  # Leave 15s buffer
        
        if mip_time_limit > 3.0:
            # Variables: x_{ij} for i in n_cust, j in n_fac (n_cust * n_fac vars)
            #             y_j for j in n_fac (n_fac vars)
            # Total vars: n_cust * n_fac + n_fac
            
            # Objective coefficients
            c = np.zeros(n_cust * n_fac + n_fac)
            c[:n_cust * n_fac] = assign_cost.flatten()
            c[n_cust * n_fac:] = open_cost
            
            # Constraints
            # 1. Assignment: sum_j x_{ij} = 1 for all i
            A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
            for i in range(n_cust):
                A_eq[i, i * n_fac : (i + 1) * n_fac] = 1.0
            b_eq = np.ones(n_cust)
            
            # 2. Capacity: sum_i demands[i] * x_{ij} <= capacities[j] * y_j for all j
            A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
            for j in range(n_fac):
                for i in range(n_cust):
                    A_ub[j, i * n_fac + j] = demands[i]
                A_ub[j, n_cust * n_fac + j] = -capacities[j]
            b_ub = np.zeros(n_fac)
            
            # Bounds
            integrality = np.ones(n_cust * n_fac + n_fac)
            lb = np.zeros(n_cust * n_fac + n_fac)
            ub = np.ones(n_cust * n_fac + n_fac)
            
            # Define constraints
            constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)
            constraint_ub = LinearConstraint(A_ub, -np.inf, b_ub)
            bounds = Bounds(lb, ub)
            
            # Warm start from current best
            x_start = np.zeros(n_cust * n_fac + n_fac)
            for i in range(n_cust):
                x_start[i * n_fac + best_result[i]] = 1.0
            for j in range(n_fac):
                if np.any(best_result == j):
                    x_start[n_cust * n_fac + j] = 1.0
            
            # Solve with time limit
            res = milp(
                c=c,
                constraints=[constraint_eq, constraint_ub],
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': mip_time_limit, 'mip_start': x_start}
            )
            
            if res.x is not None and len(res.x) == n_cust * n_fac + n_fac:
                x_mat = res.x[:n_cust * n_fac].reshape(n_cust, n_fac)
                result_mip = np.argmax(x_mat, axis=1).astype(int)
                
                if _is_valid(result_mip, demands, capacities):
                    cost_mip = _compute_cost(result_mip, demands, capacities, open_cost, assign_cost)
                    if cost_mip < best_cost:
                        best_result = result_mip
                        best_cost = cost_mip
                        result = result_mip
                    
                    # Refine MIP result with local search if time permits
                    elapsed = time.time() - start_time
                    if deadline - start_time > 10.0:
                        try:
                            refined = _refine_with_local_search(result_mip, demands, capacities, 
                                                               open_cost, assign_cost, 
                                                               time_limit=min(10.0, deadline - start_time - 5.0))
                            if _is_valid(refined, demands, capacities):
                                cost_refined = _compute_cost(refined, demands, capacities, open_cost, assign_cost)
                                if cost_refined < best_cost:
                                    best_result = refined
                                    best_cost = cost_refined
                        except Exception:
                            pass
    except Exception:
        pass
    
    # 4. Final validation
    if not _is_valid(best_result, demands, capacities):
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    return best_result

# EVOLVE-BLOCK-END
