"""Capacitated facility location: Multi-start local search with swap moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a local search strategy.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 140s max
    deadline = time.time() + 140.0
    
    # Precompute customer order by demand (largest first) for initialization stability
    # But the core algorithm is vectorized, so we'll use random restarts with different 
    # initializations.
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Initialize a valid solution using a greedy heuristic (fast)
    # This ensures we always have a valid fallback.
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    # Sort customers by demand descending for better greedy packing
    order = np.argsort(-demands)
    for i in order:
        # Find feasible facilities
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            # This shouldn't happen if total capacity >= total demand, 
            # but if it does, we have to pick the one with most remaining capacity
            # or just fail. The problem implies a valid solution exists.
            # For safety, pick the facility with max remaining capacity.
            remaining = capacities - load
            f = np.argmax(remaining)
            if load[f] + demands[i] > capacities[f]:
                # Invalid instance? Return something, though it will be invalid.
                # But the contract says valid instances.
                pass
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            # Among feasible, pick cheapest assignment cost
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    # Calculate score for greedy
    used_fac = np.unique(best_result)
    score_greedy = np.sum(open_cost[used_fac]) + np.sum(assign_cost[np.arange(n_cust), best_result])
    best_score = score_greedy

    def calc_cost(res):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def local_search(res, load):
        """Local search with single moves and swap moves."""
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            
            # Single customer moves
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                    
                f_old = res[i]
                best_delta = np.inf
                best_f_new = -1
                
                for f_new in range(n_fac):
                    if f_new == f_old:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    if load[f_old] == demands[i]:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1 and best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= demands[i]
                    load[best_f_new] += demands[i]
                    improved = True
            
            # Swap moves: try swapping two customers
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() > deadline - 2.0:
                        break
                    
                    f_i = res[i]
                    f_j = res[j]
                    if f_i == f_j:
                        continue
                    
                    # Check if swap is feasible
                    # After swap: i goes to f_j, j goes to f_i
                    # Need: load[f_j] - demands[j] + demands[i] <= capacities[f_j]
                    # and:   load[f_i] - demands[i] + demands[j] <= capacities[f_i]
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    
                    # Calculate cost delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Open cost change
                    # f_i loses i, gains j
                    # f_j loses j, gains i
                    # f_i becomes empty if load[f_i] == demands[i] and demands[j] == 0 (impossible since demands >= 5)
                    # Actually f_i becomes empty only if load[f_i] == demands[i] AND no one else there
                    # But after swap, f_i has j, so it's not empty unless demands[j]==0
                    # f_j becomes empty only if load[f_j] == demands[j] AND no one else there
                    # After swap, f_j has i, so it's not empty unless demands[i]==0
                    
                    # So open cost doesn't change unless one facility becomes completely empty
                    # But since we're swapping, both facilities still have at least one customer
                    # (i goes to f_j, j goes to f_i), so open cost delta = 0
                    delta_open = 0.0
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
                        # Perform swap
                        res[i], res[j] = res[j], res[i]
                        load[f_i] -= demands[i]
                        load[f_i] += demands[j]
                        load[f_j] -= demands[j]
                        load[f_j] += demands[i]
                        improved = True
                        
        return res, load
    
    # Multi-start with different alpha penalties
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
            
        # 1. Initial Assignment based on effective cost
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)
        
        # 2. Repair Capacity Violations
        res = init_res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
        # Repair: move customers from overloaded facilities
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
                
            f = overloaded[0]
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                break
                
            best_cust = -1
            best_g = -1
            min_penalty = np.inf
            
            for i in custs_in_f:
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g]:
                        penalty = assign_cost[i, g] - assign_cost[i, f]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_cust = i
                            best_g = g
            
            if best_cust == -1:
                break
                
            res[best_cust] = best_g
            load[f] -= demands[best_cust]
            load[best_g] += demands[best_cust]
            
        # Verify validity
        if np.all(load <= capacities):
            # Run local search on this solution
            res, load = local_search(res, load)
            score = calc_cost(res)
            if score < best_score:
                best_score = score
                best_result = res.copy()
                
    # Also run local search on the greedy solution
    current_res = best_result.copy()
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[current_res[i]] += demands[i]
    
    current_res, current_load = local_search(current_res, current_load)
    final_score = calc_cost(current_res)
    if final_score < best_score:
        best_result = current_res
        best_score = final_score

    return best_result
# EVOLVE-BLOCK-END
