"""Program using Lagrangian Relaxation with Multi-Start Local Search for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation and Local Search.
    
    Args:
    demands: array-like of customer demands
    capacities: array-like of facility capacities
    open_cost: array-like of facility opening costs
    assign_cost: 2D array of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])
    
    def greedy_init():
        """Greedy initialization: assign customers in order of demand (largest first)."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            # Sort facilities by cost for this customer
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                # If no facility has space, assign to the one with most remaining space
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    # Fallback
                    res[i] = fac_order[0]
        return res

    def local_search(res, load, max_passes=100):
        """
        Local search with relocate moves.
        Returns improved res and load.
        """
        improved = True
        passes = 0
        # We track the current res and load
        cur_res = res.copy()
        cur_load = load.copy()
        
        while improved and passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            
            # Try relocate moves for each customer
            # Random order to avoid bias
            for i in np.random.permutation(n_cust):
                current_f = cur_res[i]
                best_f = -1
                best_delta = 1e-9
                
                # Calculate potential deltas for all other facilities
                # Delta = (c[i, f_new] - c[i, current_f]) + open_cost_change
                # open_cost_change:
                #   +open_cost[f_new] if cur_load[f_new] == 0
                #   -open_cost[current_f] if cur_load[current_f] == demands[i]
                
                # Vectorized check for capacity
                # space = capacities - cur_load
                # space[current_f] is irrelevant for move TO current_f, but we skip it anyway
                # We need f_new such that cur_load[f_new] + demands[i] <= capacities[f_new]
                
                space = capacities - cur_load
                space[current_f] = -1 # Cannot move to current facility
                
                valid_f = np.where(space >= demands[i])[0]
                
                if len(valid_f) == 0:
                    continue
                
                # Calculate assignment cost delta for valid facilities
                # cost_delta = assign_cost[i, valid_f] - assign_cost[i, current_f]
                cost_delta = assign_cost[i, valid_f] - assign_cost[i, current_f]
                
                # Calculate open cost delta
                # If moving to f_new which is empty, add open_cost[f_new]
                # If moving from current_f which becomes empty, subtract open_cost[current_f]
                
                # This part is tricky to fully vectorize because the "becomes empty" condition
                # depends on the specific customer i and current_f.
                
                # However, we can pre-calculate the "base" delta.
                # Let's just loop over valid_f if the count is small, or use numpy logic.
                # n_fac is 20, so looping is fine.
                
                min_idx = 0
                min_val = np.inf
                
                for idx in range(len(valid_f)):
                    f_new = valid_f[idx]
                    delta = cost_delta[idx]
                    
                    if cur_load[f_new] == 0:
                        delta += open_cost[f_new]
                    if cur_load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    
                    if delta < min_val:
                        min_val = delta
                        min_idx = idx
                
                if min_val < 1e-9:
                    best_f = valid_f[min_idx]
                    # Apply move
                    cur_load[current_f] -= demands[i]
                    cur_load[best_f] += demands[i]
                    cur_res[i] = best_f
                    improved = True
        
        return cur_res, cur_load

    def round_to_feasible(j_i, lam):
        """
        Convert a potentially infeasible assignment to a feasible one.
        Uses effective costs guided by lam to choose moves.
        """
        res = j_i.copy()
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        
        # We will iterate until feasible or max iterations
        for _ in range(100):
            overloaded = np.where(load > capacities + 1e-9)[0]
            if len(overloaded) == 0:
                break
            
            # Pick the most overloaded facility
            f_over = overloaded[np.argmax(load[overloaded] - capacities[overloaded])]
            
            # Find a customer in f_over to move
            custs_in_f = np.where(res == f_over)[0]
            
            best_cust = -1
            best_alt = -1
            best_cost_inc = np.inf
            
            # To make this smarter, we can use the effective cost:
            # c_eff[i, f] = assign_cost[i, f] - lam[f] * demands[i]
            # We want to minimize the increase in the Lagrangian objective or just the raw cost?
            # Minimizing raw cost is safer for feasibility, but lam guides us.
            # Let's use a weighted cost: assign_cost + small penalty for lam deviation?
            # Actually, just using assign_cost for the move decision is standard for repair.
            
            for i in custs_in_f:
                space = capacities - load
                space[f_over] = -1
                valid_f = np.where(space >= demands[i])[0]
                
                if len(valid_f) == 0:
                    continue
                
                # Choose the valid facility with lowest assignment cost
                # We could also consider if opening a new facility is costly.
                # If load[f_alt] == 0, we incur open_cost[f_alt].
                
                min_cost = np.inf
                best_f_alt = -1
                
                for f_alt in valid_f:
                    cost_inc = assign_cost[i, f_alt] - assign_cost[i, f_over]
                    if load[f_alt] == 0:
                        cost_inc += open_cost[f_alt]
                    # If f_over becomes empty, we save open_cost[f_over]
                    # We can't know if it becomes empty until we move, but we can check
                    # if demands[i] == load[f_over].
                    if load[f_over] == demands[i]:
                        cost_inc -= open_cost[f_over]
                        
                    if cost_inc < min_cost:
                        min_cost = cost_inc
                        best_f_alt = f_alt
                
                if best_f_alt != -1 and min_cost < best_cost_inc:
                    best_cost_inc = min_cost
                    best_cust = i
                    best_alt = best_f_alt
            
            if best_cust == -1:
                # If we can't fix it by moving one customer, try moving the largest demand customer
                # to the facility with most space, ignoring cost (emergency)
                # This should rarely happen if total capacity > total demand
                f_over = overloaded[0]
                custs_in_f = np.where(res == f_over)[0]
                # Sort customers by demand descending
                custs_in_f = custs_in_f[np.argsort(-demands[custs_in_f])]
                for i in custs_in_f:
                    space = capacities - load
                    space[f_over] = -1
                    valid_f = np.where(space >= demands[i])[0]
                    if len(valid_f) > 0:
                        # Pick one with most space to be safe
                        best_alt = valid_f[np.argmax(space[valid_f])]
                        res[i] = best_alt
                        load[f_over] -= demands[i]
                        load[best_alt] += demands[i]
                        break
                else:
                    break
            else:
                res[best_cust] = best_alt
                load[f_over] -= demands[best_cust]
                load[best_alt] += demands[best_cust]
        
        return res

    # 1. Initial Greedy Solution
    res_greedy = greedy_init()
    load_greedy = np.zeros(n_fac)
    np.add.at(load_greedy, res_greedy, demands)
    
    # Polish the greedy solution
    res_greedy, load_greedy = local_search(res_greedy, load_greedy, max_passes=50)
    best_score = compute_score(res_greedy)
    best_assign = res_greedy.copy()
    
    # 2. Lagrangian Relaxation Loop
    lam = np.zeros(n_fac)
    max_iter = 10000
    step_scale = 1.0
    best_lb = -np.inf
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute Lagrangian value and subproblem solution
        c_eff = assign_cost - np.outer(demands, lam)
        j_i = np.argmin(c_eff, axis=1)
        
        used_fac = np.zeros(n_fac, dtype=bool)
        used_fac[j_i] = True
        
        term1 = np.dot(lam, capacities)
        term2 = np.sum(open_cost[used_fac])
        term3 = np.sum(c_eff[np.arange(n_cust), j_i])
        value = term1 + term2 + term3
        
        if value > best_lb:
            best_lb = value
            
        # Round to feasible solution
        feasible_res = round_to_feasible(j_i, lam)
        load_check = np.zeros(n_fac)
        np.add.at(load_check, feasible_res, demands)
        
        if not np.any(load_check > capacities + 1e-9):
            # Run a limited local search on this new candidate
            # We limit passes to keep it fast, as we do this every iteration
            feasible_res, load_check = local_search(feasible_res, load_check, max_passes=5)
            score = compute_score(feasible_res)
            
            if score < best_score:
                best_score = score
                best_assign = feasible_res.copy()
                # Update best solution in the pool if we had one, but here we just track single best
                
        # Compute subgradient
        load_lam = np.zeros(n_fac)
        np.add.at(load_lam, j_i, demands)
        grad = capacities - load_lam
        
        # Check convergence
        if np.allclose(grad, 0, atol=1e-6):
            break
        
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-10:
            break
        
        # Step size: (UB - L_k) / ||grad||^2
        # Use best_score as UB
        step = step_scale * (best_score - value) / grad_norm_sq
        if step < 0:
            step = 0.01
            
        # Adaptive step decay
        if it % 100 == 0 and it > 0:
            step_scale *= 0.95
            
        lam = lam + step * grad
        lam = np.maximum(lam, 0)
        
        # Periodically polish the best solution found so far
        if it % 100 == 0:
            load_best = np.zeros(n_fac)
            np.add.at(load_best, best_assign, demands)
            best_assign, load_best = local_search(best_assign, load_best, max_passes=10)
            new_score = compute_score(best_assign)
            if new_score < best_score:
                best_score = new_score

    # 3. Final Polishing
    # Use remaining time to run a deep local search on the best solution
    load_final = np.zeros(n_fac)
    np.add.at(load_final, best_assign, demands)
    
    # Run as many passes as possible until time runs out
    # We pass a large max_passes, but the function checks time
    best_assign, load_final = local_search(best_assign, load_final, max_passes=10000)
    
    # Verify feasibility
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        # Fallback to greedy if somehow infeasible (should not happen)
        best_assign = greedy_init()
    
    return best_assign.astype(int)

# EVOLVE-BLOCK-END
