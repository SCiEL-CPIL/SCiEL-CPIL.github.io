"""Capacitated facility location: Vectorized local search with close-facility moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using vectorized local search with
    single moves, swap moves, and close-facility moves.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Safety margin: 150s limit, aim for 135s max to leave buffer
    deadline = time.time() + 135.0
    
    best_score = np.inf
    best_result = np.zeros(n_cust, dtype=int)
    
    # Initialize a valid solution using a greedy heuristic (fast)
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            remaining = capacities - load
            f = np.argmax(remaining)
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = feasible[np.argmin(costs)]
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]
            
    best_result = result_greedy.copy()
    used_fac = np.unique(best_result)
    score_greedy = np.sum(open_cost[used_fac]) + np.sum(assign_cost[np.arange(n_cust), best_result])
    best_score = score_greedy

    def calc_cost(res):
        used = np.unique(res)
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def local_search(res, load):
        """
        Local search with vectorized single moves, swap moves, and close-facility moves.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            
            # === Vectorized single customer moves ===
            # For each customer i, find the best facility to move to
            # delta = assign_cost[i, f_new] - assign_cost[i, f_old] + open_cost_delta
            # open_cost_delta = -open_cost[f_old] if load[f_old]==demands[i] else 0
            #                  + open_cost[f_new] if load[f_new]==0 else 0
            
            # Precompute base costs: assign_cost[i, res[i]] for all i
            base_assign = assign_cost[np.arange(n_cust), res]
            
            # For each customer, compute delta for all possible target facilities
            # Feasibility: load[f_new] + demands[i] <= capacities[f_new]
            # We'll iterate customers but vectorize over facilities
            
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                
                f_old = res[i]
                d_i = demands[i]
                
                # Feasible targets: capacity allows
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[f_old] = False  # Exclude current facility
                
                if not np.any(feasible_mask):
                    continue
                
                # Delta assignment cost for all feasible facilities
                delta_assign = assign_cost[i, :] - base_assign[i]
                
                # Open cost delta:
                # -open_cost[f_old] if this customer is the only one at f_old
                # +open_cost[f_new] if f_new is currently empty
                close_old = (load[f_old] == d_i)
                open_new = (load == 0)
                
                delta_open = np.zeros(n_fac)
                if close_old:
                    delta_open[f_old] = -open_cost[f_old]
                if np.any(open_new):
                    delta_open[open_new] = open_cost[open_new]
                
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                
                if best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    improved = True
            
            # === Close-facility move: try to close a facility by relocating all its customers ===
            if time.time() > deadline - 2.0:
                break
            
            for f_close in range(n_fac):
                if time.time() > deadline - 2.0:
                    break
                
                if load[f_close] == 0:
                    continue
                
                # Only consider closing if it saves open cost and we can relocate everyone
                custs = np.where(res == f_close)[0]
                if len(custs) == 0:
                    continue
                
                # Try to find a valid reassignment for all customers in f_close
                # Use a simple greedy: for each customer, find best feasible facility
                # that doesn't overload, preferring already-open facilities
                
                # Build a temporary assignment
                temp_load = load.copy()
                temp_load[f_close] = 0  # Free up capacity at f_close
                temp_res = res.copy()
                
                # Sort customers by demand descending for better packing
                cust_order = custs[np.argsort(-demands[custs])]
                
                success = True
                for c in cust_order:
                    d_c = demands[c]
                    feasible = np.where(temp_load + d_c <= capacities)[0]
                    if len(feasible) == 0:
                        success = False
                        break
                    
                    # Prefer facilities that are already open (have load > 0) to avoid new open costs
                    # Among feasible, pick the one with min assign_cost
                    costs = assign_cost[c, feasible]
                    # Add penalty for opening new facilities
                    open_penalty = np.where(temp_load[feasible] == 0, open_cost[feasible], 0.0)
                    adjusted_costs = costs + open_penalty
                    best_f = feasible[np.argmin(adjusted_costs)]
                    
                    temp_res[c] = best_f
                    temp_load[best_f] += d_c
                
                if not success:
                    continue
                
                # Check if all loads are within capacity
                if np.any(temp_load > capacities):
                    continue
                
                # Compute the score change
                # Old: open_cost[f_close] (if f_close was the only user) + sum of assign costs for custs at f_close
                # New: sum of assign costs for custs at their new facilities + any new open costs
                
                # Check if f_close becomes empty (it should by construction)
                if temp_load[f_close] > 0:
                    continue  # Shouldn't happen but safety check
                
                # Calculate total cost difference
                old_assign_sum = np.sum(assign_cost[custs, f_close])
                new_assign_sum = np.sum(assign_cost[custs, temp_res[custs]])
                
                # Open cost change:
                # Remove f_close's open cost
                # Add open cost for any newly opened facilities
                newly_opened = set()
                for c in custs:
                    nf = temp_res[c]
                    if nf != f_close and temp_load[nf] > 0:
                        # Check if this facility was empty before the move
                        # We need to check original load
                        if load[nf] == 0:
                            newly_opened.add(nf)
                
                open_cost_change = -open_cost[f_close]
                for nf in newly_opened:
                    open_cost_change += open_cost[nf]
                
                total_delta = (new_assign_sum - old_assign_sum) + open_cost_change
                
                if total_delta < -1e-9:
                    # Accept the close-facility move
                    for c in custs:
                        res[c] = temp_res[c]
                    load = temp_load.copy()
                    improved = True
            
            # === Swap moves: try swapping two customers ===
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
                    
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Open cost change for swaps:
                    # If f_i becomes empty after swap: -open_cost[f_i]
                    # If f_j becomes empty after swap: -open_cost[f_j]
                    # If f_i was empty before and gets customer j: +open_cost[f_i] (but f_i has customer i, so not empty)
                    # If f_j was empty before and gets customer i: +open_cost[f_j] (but f_j has customer j, so not empty)
                    # So open cost only changes if a facility becomes empty
                    delta_open = 0.0
                    if load[f_i] - demands[i] + demands[j] == 0:
                        delta_open -= open_cost[f_i]
                    if load[f_j] - demands[j] + demands[i] == 0:
                        delta_open -= open_cost[f_j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9:
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
            
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1)
        
        res = init_res.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[res[i]] += demands[i]
            
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
            
        if np.all(load <= capacities):
            used = np.unique(res)
            score = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
            if score < best_score:
                best_score = score
                best_result = res.copy()
                
                # Run local search on the best result
                res_ls, load_ls = local_search(best_result.copy(), load.copy())
                if np.all(load_ls <= capacities):
                    used_ls = np.unique(res_ls)
                    score_ls = np.sum(open_cost[used_ls]) + np.sum(assign_cost[np.arange(n_cust), res_ls])
                    if score_ls < best_score:
                        best_score = score_ls
                        best_result = res_ls.copy()

    # Final local search on the best result found
    current_load = np.zeros(n_fac)
    for i in range(n_cust):
        current_load[best_result[i]] += demands[i]
    
    res_final, load_final = local_search(best_result.copy(), current_load.copy())
    if np.all(load_final <= capacities):
        used_final = np.unique(res_final)
        score_final = np.sum(open_cost[used_final]) + np.sum(assign_cost[np.arange(n_cust), res_final])
        if score_final < best_score:
            best_result = res_final
            best_score = score_final

    return best_result
# EVOLVE-BLOCK-END
