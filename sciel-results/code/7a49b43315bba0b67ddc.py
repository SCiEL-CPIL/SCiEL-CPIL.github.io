"""
Capacitated Facility Location: Lagrangian Relaxation with Subgradient Optimization 
and Local Search Polish.
"""
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
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Initial best solution
    best_assignment = None
    best_cost = float('inf')

    def compute_cost(assignment):
        """Compute the true objective value for a feasible assignment."""
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(assignment):
            if f < 0 or f >= n_fac:
                return float('inf')
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-6):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), assignment])

    def repair_and_polish(assignment, time_limit):
        """
        Repair infeasible assignment and perform local search.
        Returns the best feasible assignment found within time_limit.
        """
        assignment = assignment.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(assignment):
            load[f] += demands[i]
            fac_count[f] += 1
            
        # Repair phase
        max_repair_iters = 1000
        for _ in range(max_repair_iters):
            overloaded = np.where(load > capacities + 1e-6)[0]
            if len(overloaded) == 0:
                break
            
            # Find the most overloaded facility
            # Or just pick the first one
            f_from = overloaded[0]
            
            # Find a customer to move from f_from
            # Prioritize customers that can be moved to a facility with the least cost increase
            # and where the target facility has the most spare capacity
            best_move_i = -1
            best_move_to = -1
            min_penalty = float('inf')
            
            customers_in_f = np.where(assignment == f_from)[0]
            for i in customers_in_f:
                d_i = demands[i]
                for f_to in range(n_fac):
                    if f_to == f_from:
                        continue
                    if load[f_to] + d_i <= capacities[f_to] + 1e-6:
                        # Penalty is the increase in assignment cost
                        # We ignore open cost changes for simplicity in repair, 
                        # assuming open costs are handled by local search
                        penalty = assign_cost[i, f_to] - assign_cost[i, f_from]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_move_i = i
                            best_move_to = f_to
            
            if best_move_i == -1:
                # No feasible move found? Try moving to any facility with space, even if cost increases a lot
                # Or if truly stuck, we might have an infeasible instance (shouldn't happen)
                # For now, break
                break
                
            # Perform move
            i = best_move_i
            f_to = best_move_to
            assignment[i] = f_to
            load[f_from] -= demands[i]
            load[f_to] += demands[i]
            fac_count[f_from] -= 1
            fac_count[f_to] += 1
        
        # Check if feasible
        if np.any(load > capacities + 1e-6):
            return None, float('inf')
            
        current_cost = compute_cost(assignment)
        
        # Local Search Phase
        if current_cost < float('inf'):
            current = assignment.copy()
            curr_load = load.copy()
            curr_fac_count = fac_count.copy()
            curr_cost = current_cost
            best_local_cost = curr_cost
            best_local_assignment = current.copy()
            
            start_ls = time.time()
            improved = True
            no_improve_count = 0
            
            while improved and (time.time() - start_ls < time_limit):
                improved = False
                no_improve_count = 0
                
                # 1. Single Reassignment (1-move)
                # Shuffle customer order to avoid bias
                order = np.random.permutation(n_cust)
                for i in order:
                    if time.time() - start_ls > time_limit:
                        break
                    cur_f = current[i]
                    d_i = demands[i]
                    best_delta = 0
                    best_new_f = -1
                    
                    for new_f in range(n_fac):
                        if new_f == cur_f:
                            continue
                        if curr_load[new_f] + d_i > capacities[new_f] + 1e-6:
                            continue
                        
                        delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
                        
                        if curr_fac_count[cur_f] == 1:
                            delta -= open_cost[cur_f]
                        if curr_fac_count[new_f] == 0:
                            delta += open_cost[new_f]
                            
                        if delta < best_delta:
                            best_delta = delta
                            best_new_f = new_f
                            
                    if best_new_f != -1:
                        old_f = cur_f
                        new_f = best_new_f
                        current[i] = new_f
                        curr_load[old_f] -= d_i
                        curr_load[new_f] += d_i
                        curr_fac_count[old_f] -= 1
                        curr_fac_count[new_f] += 1
                        curr_cost += best_delta
                        improved = True
                        if curr_cost < best_local_cost:
                            best_local_cost = curr_cost
                            best_local_assignment = current.copy()
                        # Continue searching for more improvements in this pass
                        # But to be faster, we can restart the pass if we found a good one
                        # Let's just continue to the next customer
                        break # Restart pass to get fresh state? Or just continue? 
                        # Continuing is fine, but restarting ensures we don't miss better moves 
                        # due to the changed state. However, for speed, let's just continue 
                        # and rely on multiple passes.
                        # Actually, breaking and restarting is safer for convergence.
                        # But it's slower. Let's just continue.
                        
                # 2. Swap (2-move)
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > 5:
                        break
                        
                order = np.random.permutation(n_cust)
                for i in order:
                    if time.time() - start_ls > time_limit:
                        break
                    f_i = current[i]
                    d_i = demands[i]
                    for j in range(i + 1, n_cust):
                        if time.time() - start_ls > time_limit:
                            break
                        f_j = current[j]
                        if f_i == f_j:
                            continue
                        d_j = demands[j]
                        
                        if curr_load[f_i] - d_i + d_j > capacities[f_i] + 1e-6:
                            continue
                        if curr_load[f_j] - d_j + d_i > capacities[f_j] + 1e-6:
                            continue
                        
                        delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                        
                        # Open costs don't change in a swap between non-empty facilities
                        # If a facility becomes empty, we save its open cost.
                        # If a facility becomes non-empty (was empty), we pay its open cost.
                        # In a swap, if f_i and f_j are distinct:
                        # Count of f_i: -1 (i leaves) + 1 (j enters) = 0 change.
                        # Count of f_j: -1 (j leaves) + 1 (i enters) = 0 change.
                        # So open costs are invariant.
                        
                        if delta < 0:
                            current[i] = f_j
                            current[j] = f_i
                            curr_load[f_i] = curr_load[f_i] - d_i + d_j
                            curr_load[f_j] = curr_load[f_j] - d_j + d_i
                            curr_cost += delta
                            improved = True
                            if curr_cost < best_local_cost:
                                best_local_cost = curr_cost
                                best_local_assignment = current.copy()
                            break # Restart swap search? Or just break inner loop?
                            # Breaking inner loop means we try next i.
                        
            return best_local_assignment, best_local_cost

    # --- Lagrangian Relaxation ---
    
    # Multipliers for capacity constraints
    lambda_ = np.zeros(n_fac)
    
    # Step size parameters
    step_size = 1.0
    max_iterations = 2000
    
    # Initial solution: Greedy
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
        init_cost = compute_cost(init_res)
        if init_cost < best_cost:
            best_cost = init_cost
            best_assignment = init_res.copy()

    # Lagrangian Loop
    for iter in range(max_iterations):
        if time.time() > deadline - 20.0: # Leave 20s for final polish
            break
            
        # Solve subproblem
        # Adjusted cost for customer i to facility j: assign_cost[i, j] + lambda_[j] * demands[i]
        # Note: The Lagrangian function is L(x, lambda) = sum c_ij x_ij + sum lambda_j (sum d_i x_ij - C_j)
        # = sum (c_ij + lambda_j d_i) x_ij - sum lambda_j C_j
        # We minimize the first term. The constant term doesn't affect the argmin.
        
        # Vectorized solution for subproblem
        # adjusted_costs[i, j] = assign_cost[i, j] + lambda_[j] * demands[i]
        adjusted_costs = assign_cost + np.outer(demands, lambda_)
        
        # For each customer, pick the facility with min adjusted cost
        sub_assignment = np.argmin(adjusted_costs, axis=1)
        
        # Calculate load in subproblem
        sub_load = np.zeros(n_fac)
        np.add.at(sub_load, sub_assignment, demands)
        
        # Calculate Lagrangian lower bound
        # LB = sum_i min_j (c_ij + lambda_j d_i) - sum_j lambda_j C_j
        min_costs = np.min(adjusted_costs, axis=1)
        lb = np.sum(min_costs) - np.dot(lambda_, capacities)
        
        # Update best feasible solution if we can repair it quickly?
        # No, repair is expensive. We just use the subproblem solution as a hint 
        # or try to repair it every K iterations.
        
        # Update multipliers using subgradient
        # g_j = sum_i d_i x_ij - C_j
        subgrad = sub_load - capacities
        
        # Polyak step size if we have a good upper bound (UB)
        # step = (UB - LB) / ||g||^2
        if best_cost < float('inf'):
            norm_sq = np.sum(subgrad ** 2)
            if norm_sq > 1e-10:
                step_size = (best_cost - lb) / norm_sq
                if step_size < 0:
                    step_size = 0
            else:
                step_size = 0
        else:
            # Adaptive step size
            step_size = 1.0 / (1 + iter)
            
        # Damping
        step_size = min(step_size, 2.0) # Cap step size
        
        lambda_ = lambda_ + step_size * subgrad
        lambda_ = np.maximum(lambda_, 0) # Multipliers must be non-negative
        
        # Occasionally try to repair the subproblem solution
        if iter % 10 == 0:
            repaired, cost = repair_and_polish(sub_assignment, 1.0)
            if repaired is not None and cost < best_cost:
                best_cost = cost
                best_assignment = repaired

    # Final Polish with remaining time
    time_remaining = deadline - time.time() - 2.0
    if time_remaining > 1.0 and best_assignment is not None:
        polished, cost = repair_and_polish(best_assignment, time_remaining)
        if polished is not None and cost < best_cost:
            best_assignment = polished

    if best_assignment is None:
        # Last resort
        best_assignment = greedy_init()
        if best_assignment is None:
            best_assignment = np.zeros(n_cust, dtype=int)

    return best_assignment.astype(int)
