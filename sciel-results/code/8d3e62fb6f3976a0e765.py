"""Capacitated facility location: Lagrangian Relaxation + Local Search."""
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Lagrangian Relaxation and Local Search.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_true_cost(result):
        """Compute true cost for a feasible assignment."""
        if result is None:
            return float('inf')
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-6):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def repair_assignment(result, load):
        """
        Repair an infeasible assignment by moving customers from overloaded facilities.
        Returns a feasible assignment and its load.
        """
        result = result.astype(int).copy()
        load = load.copy()
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            fac_count[f] += 1
            
        # Identify overloaded facilities
        overloaded = np.where(load > capacities + 1e-6)[0]
        
        # If no overloaded, it's feasible
        if len(overloaded) == 0:
            return result, load, fac_count

        # We need to move customers out of overloaded facilities
        # Sort customers in overloaded facilities by "cost of moving" to next best facility
        # We'll iterate until feasible or no moves possible
        
        max_iterations = 100
        for _ in range(max_iterations):
            overloaded = np.where(load > capacities + 1e-6)[0]
            if len(overloaded) == 0:
                break
            
            # Find a customer to move from an overloaded facility
            # Prioritize customers with the smallest penalty for moving
            # Penalty = min_cost_to_other_facility - current_cost
            
            best_move_customer = -1
            best_move_from = -1
            best_move_to = -1
            min_penalty = float('inf')
            
            for f_from in overloaded:
                # Get customers assigned to f_from
                customers_in_f = np.where(result == f_from)[0]
                for i in customers_in_f:
                    # Find best alternative facility
                    # Must have capacity
                    # Cost = assign_cost[i, f_to] - assign_cost[i, f_from]
                    # We want to minimize the increase in cost
                    # Also, we need to ensure f_to has space
                    feasible_to = []
                    for f_to in range(n_fac):
                        if f_to == f_from:
                            continue
                        if load[f_to] + demands[i] <= capacities[f_to] + 1e-6:
                            # Calculate penalty
                            # If f_to is currently closed, we pay open_cost
                            # If f_from becomes empty, we save open_cost
                            # For simplicity in repair, we just look at assignment cost difference
                            # and assume open costs are handled by the final cost calculation
                            # But to be safe, let's just pick the one with lowest assignment cost increase
                            penalty = assign_cost[i, f_to] - assign_cost[i, f_from]
                            feasible_to.append((penalty, f_to))
                    
                    if feasible_to:
                        best_penalty, best_f_to = min(feasible_to, key=lambda x: x[0])
                        if best_penalty < min_penalty:
                            min_penalty = best_penalty
                            best_move_customer = i
                            best_move_from = f_from
                            best_move_to = best_f_to
            
            if best_move_customer == -1:
                # No feasible move found? This should not happen if total capacity > total demand
                # If it does, we have a problem. Return infeasible or best effort.
                break
            
            # Perform the move
            i = best_move_customer
            f_from = best_move_from
            f_to = best_move_to
            
            result[i] = f_to
            load[f_from] -= demands[i]
            load[f_to] += demands[i]
            fac_count[f_from] -= 1
            fac_count[f_to] += 1
            
        return result, load, fac_count

    def local_search(result, time_limit):
        """Local search with 1-moves and 2-swaps."""
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1
            
        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])
            
        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()
        
        start = time.time()
        improved = True
        
        while improved and (time.time() - start < time_limit):
            improved = False
            
            # 1. Single Reassignment (1-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1
                
                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f] + 1e-6:
                        continue
                    
                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                    
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f
                        
                if best_new_f != -1:
                    old_f = cur_f
                    new_f = best_new_f
                    current[i] = new_f
                    load[old_f] -= d_i
                    load[new_f] += d_i
                    fac_count[old_f] -= 1
                    fac_count[new_f] += 1
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()
            
            # 2. Swap (2-move)
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    if load[f_i] - d_i + d_j > capacities[f_i] + 1e-6:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j] + 1e-6:
                        continue
                    
                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if delta < 0:
                        current[i] = f_j
                        current[j] = f_i
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break
            
        return best_result

    # --- Lagrangian Relaxation Phase ---
    lambda_vec = np.zeros(n_fac)
    best_feasible_result = None
    best_feasible_cost = float('inf')
    
    # Initial greedy solution to provide an Upper Bound (UB)
    def greedy_init():
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-6:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                # Fallback: find any feasible
                feasible_facs = np.where(capacities - load >= demands[i])[0]
                if len(feasible_facs) > 0:
                    best_f = min(feasible_facs, key=lambda f: assign_cost[i, f])
                    load[best_f] += demands[i]
                    result[i] = best_f
                else:
                    return None
        return result

    init_res = greedy_init()
    if init_res is not None:
        init_cost = compute_true_cost(init_res)
        if init_cost < best_feasible_cost:
            best_feasible_cost = init_cost
            best_feasible_result = init_res.copy()
    
    # Lagrangian Loop
    max_iter = 500
    damping = 0.5
    for k in range(max_iter):
        if time.time() > deadline - 60: # Reserve 60s for local search
            break
        
        # 1. Solve subproblem: x_ij = 1 if j = argmin (a_ij + d_i lambda_j)
        # Cost matrix M = assign_cost + demands[:, None] * lambda_vec[None, :]
        M = assign_cost + demands[:, None] * lambda_vec[None, :]
        best_fac_idx = np.argmin(M, axis=1)
        
        # Build assignment
        x = np.zeros((n_cust, n_fac))
        for i in range(n_cust):
            x[i, best_fac_idx[i]] = 1.0
            
        # Compute load
        load = np.sum(x * demands[:, None], axis=0)
        
        # Determine open facilities y
        y = (load > 1e-6).astype(float)
        
        # Compute Lagrangian Value
        # L(lambda) = sum_j (o_j - lambda_j C_j) y_j + sum_ij (a_ij + d_i lambda_j) x_ij
        # Note: sum_ij (a_ij + d_i lambda_j) x_ij = sum_i M[i, best_fac_idx[i]]
        assign_part = np.sum(M[np.arange(n_cust), best_fac_idx])
        open_part = np.sum((open_cost - lambda_vec * capacities) * y)
        L_val = assign_part + open_part
        
        # Compute subgradient g_j = load_j - C_j y_j
        g = load - capacities * y
        
        # Update lambda
        # Step size alpha
        norm_g_sq = np.sum(g**2)
        if norm_g_sq < 1e-12:
            break
            
        if best_feasible_cost < float('inf'):
            # Polyak step size
            alpha = (best_feasible_cost - L_val) / norm_g_sq * damping
            if alpha < 0:
                alpha = 0.0
            # Cap alpha to prevent huge jumps
            max_alpha = 10.0
            if alpha > max_alpha:
                alpha = max_alpha
        else:
            # Fixed step if no UB
            alpha = 0.1
            
        lambda_vec = np.maximum(0, lambda_vec + alpha * g)
        
        # Repair and Check Feasibility
        # Current x is an assignment (each customer to one facility)
        # Check capacity
        current_result = best_fac_idx.astype(int)
        is_feasible = np.all(load <= capacities + 1e-6)
        
        if is_feasible:
            true_cost = compute_true_cost(current_result)
            if true_cost < best_feasible_cost:
                best_feasible_cost = true_cost
                best_feasible_result = current_result.copy()
        else:
            # Repair
            repaired, rep_load, _ = repair_assignment(current_result, load)
            rep_cost = compute_true_cost(repaired)
            if rep_cost < best_feasible_cost:
                best_feasible_cost = rep_cost
                best_feasible_result = repaired.copy()
                
        # Decay damping
        damping *= 0.99

    # --- Local Search Phase ---
    if best_feasible_result is not None:
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed - 5.0 # Safety margin
        
        if time_remaining > 1.0:
            polished = local_search(best_feasible_result, time_remaining)
            if polished is not None:
                polished_cost = compute_true_cost(polished)
                if polished_cost < best_feasible_cost:
                    best_feasible_result = polished
    
    if best_feasible_result is None:
        # Fallback to simple greedy if something went wrong
        best_feasible_result = greedy_init()
        if best_feasible_result is None:
             # Last resort: assign to facility 0 (might be infeasible, but better than crash)
             best_feasible_result = np.zeros(n_cust, dtype=int)
             
    return best_feasible_result
