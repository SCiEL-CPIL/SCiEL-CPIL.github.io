"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute sorted facility indices by cost for each customer
    # sorted_fac_by_cost[i] is an array of facility indices for customer i, sorted by assign_cost[i, f]
    sorted_fac_by_cost = np.argsort(assign_cost, axis=1)
    
    def calc_cost(assignment, load, open_flags):
        # assignment: array of facility indices
        # load: array of loads
        # open_flags: boolean array
        assign_sum = np.sum(assign_cost[np.arange(n_cust), assignment])
        open_sum = np.sum(open_cost[open_flags])
        return assign_sum + open_sum

    def is_feasible(assignment, load):
        return np.all(load <= capacities)

    def local_search(assignment, load, open_flags, total_cost, max_time=10.0):
        """
        Performs best-improvement local search using Relocate and Swap moves.
        Returns updated assignment, load, open_flags, total_cost.
        """
        start_ls = time.time()
        improved = True
        # Precompute some things for speed?
        # We need to check capacity often.
        
        while improved:
            if time.time() - start_ls > max_time:
                break
            improved = False
            best_delta = 1e-9
            best_move = None
            best_type = None # 'relocate' or 'swap'
            
            # 1. Relocate Moves
            for i in range(n_cust):
                f_old = assignment[i]
                d_i = demands[i]
                c_i_old = assign_cost[i, f_old]
                
                # Try all other facilities
                # Optimization: iterate through sorted costs for this customer
                for f_new in sorted_fac_by_cost[i]:
                    if f_new == f_old:
                        continue
                    
                    # Check capacity
                    if load[f_new] + d_i > capacities[f_new]:
                        continue
                    
                    # Calculate delta
                    delta_assign = assign_cost[i, f_new] - c_i_old
                    
                    delta_open = 0.0
                    # Check if f_old becomes empty
                    if load[f_old] == d_i:
                        delta_open -= open_cost[f_old]
                    # Check if f_new was empty
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    delta_total = delta_assign + delta_open
                    
                    if delta_total < best_delta:
                        best_delta = delta_total
                        best_move = (i, f_old, f_new)
                        best_type = 'relocate'
            
            # 2. Swap Moves
            # Only consider swaps if we haven't found a very good relocate move? 
            # Or just check all. N^2 is 10k, which is small.
            for i in range(n_cust):
                f_i = assignment[i]
                d_i = demands[i]
                c_i_fi = assign_cost[i, f_i]
                
                for j in range(i + 1, n_cust):
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    c_j_fj = assign_cost[j, f_j]
                    
                    # Check capacities for swap
                    # f_i loses d_i, gains d_j. New load: load[f_i] - d_i + d_j
                    # f_j loses d_j, gains d_i. New load: load[f_j] - d_j + d_i
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    
                    # Calculate delta
                    # Cost change for i: c[i, f_j] - c[i, f_i]
                    # Cost change for j: c[j, f_i] - c[j, f_j]
                    delta_assign = (assign_cost[i, f_j] - c_i_fi) + (assign_cost[j, f_i] - c_j_fj)
                    
                    # Open cost change?
                    # Since both facilities are currently open (have at least one customer), 
                    # and after swap they still have at least one customer (unless one was size 1 and we swap? 
                    # No, if f_i had only i, and we move i to f_j, f_i becomes empty. 
                    # But j moves to f_i. So f_i has j. So f_i is not empty.
                    # Same for f_j.
                    # So open cost does not change for swaps between non-empty facilities.
                    delta_open = 0.0
                    
                    delta_total = delta_assign + delta_open
                    
                    if delta_total < best_delta:
                        best_delta = delta_total
                        best_move = (i, j, f_i, f_j)
                        best_type = 'swap'
            
            # Apply best move
            if best_move is not None:
                if best_type == 'relocate':
                    i, f_old, f_new = best_move
                    d_i = demands[i]
                    
                    # Update load
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    
                    # Update open flags
                    if load[f_old] == 0:
                        open_flags[f_old] = False
                    if load[f_new] == 0:
                        open_flags[f_new] = True # Shouldn't happen if d_i > 0 and it was empty before? 
                                                 # If it was empty, load was 0. Now load is d_i > 0. So it becomes open.
                                                 # Wait, if load[f_new] was 0, it was NOT open. Now it is.
                                                 # My delta_open logic: if load[f_new]==0 (before), delta_open += open_cost.
                                                 # This matches.
                    
                    # Update assignment
                    assignment[i] = f_new
                    
                    total_cost += best_delta
                    improved = True
                    
                elif best_type == 'swap':
                    i, j, f_i, f_j = best_move
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    # Update load
                    load[f_i] = load[f_i] - d_i + d_j
                    load[f_j] = load[f_j] - d_j + d_i
                    
                    # Update assignment
                    assignment[i] = f_j
                    assignment[j] = f_i
                    
                    total_cost += best_delta
                    improved = True
        
        return assignment, load, open_flags, total_cost

    def greedy_init(demands, capacities, assign_cost, sorted_fac_by_cost, mode='demand'):
        """
        Greedy initialization.
        mode='demand': Sort customers by demand descending.
        mode='cost': Sort customers by min cost ascending.
        """
        assignment = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        open_flags = np.zeros(n_fac, dtype=bool)
        
        if mode == 'demand':
            order = np.argsort(-demands)
        elif mode == 'cost':
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
        else:
            order = np.arange(n_cust)
            
        for i in order:
            d_i = demands[i]
            assigned = False
            for f in sorted_fac_by_cost[i]:
                if load[f] + d_i <= capacities[f]:
                    assignment[i] = f
                    load[f] += d_i
                    if load[f] > 0:
                        open_flags[f] = True
                    assigned = True
                    break
            if not assigned:
                # Should not happen if total demand <= total capacity
                # But if it does, we need to handle it. 
                # For now, just pick any facility with space? 
                # If no space, the instance is infeasible? 
                # Problem statement implies feasible instances.
                # Let's just break or pick first feasible.
                # If we reach here, we failed.
                pass
                
        return assignment, load, open_flags

    def lagrangian_guided_init(demands, capacities, assign_cost, sorted_fac_by_cost, open_cost):
        """
        Uses Lagrangian relaxation to guide the greedy assignment.
        """
        # 1. Find multipliers
        lambdas = np.zeros(n_fac)
        # Simple subgradient method
        # We want to balance load.
        # Target: load_f <= cap_f
        # Penalty: lambda_f * (load_f - cap_f)
        
        # Run for a few iterations to get a "good" lambda set
        # This is a heuristic, so we don't need perfect convergence.
        step = 1.0
        for _ in range(50):
            # Relaxed assignment: min (c_if + lambda_f * d_i)
            # Note: This ignores open cost, which is okay for initialization guidance.
            eff_costs = assign_cost + np.outer(demands, lambdas)
            relaxed_assign = np.argmin(eff_costs, axis=1)
            
            load = np.bincount(relaxed_assign, weights=demands, minlength=n_fac)
            
            # Check if feasible
            if np.all(load <= capacities):
                break
            
            # Update lambdas
            # Increase lambda for overloaded, decrease for underloaded (but keep >= 0)
            subgrad = load - capacities
            lambdas = np.maximum(0, lambdas + step * subgrad)
            step *= 0.9
            
        # 2. Greedy construction guided by lambdas
        # Effective cost for sorting: c_if + lambda_f * d_i
        # We create a dynamic sort for each customer based on these effective costs.
        # Since lambdas are fixed now, we can precompute the order.
        
        assignment = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        open_flags = np.zeros(n_fac, dtype=bool)
        
        # Sort customers by demand descending to handle big ones first
        order = np.argsort(-demands)
        
        for i in order:
            d_i = demands[i]
            # Calculate effective costs for this customer with current lambdas
            # eff_cost_f = assign_cost[i, f] + lambdas[f] * d_i
            # We want to try facilities in order of increasing eff_cost.
            # We can just compute the argsort for this specific row.
            eff_costs_row = assign_cost[i, :] + lambdas * d_i
            sorted_facs = np.argsort(eff_costs_row)
            
            assigned = False
            for f in sorted_facs:
                if load[f] + d_i <= capacities[f]:
                    assignment[i] = f
                    load[f] += d_i
                    if load[f] > 0:
                        open_flags[f] = True
                    assigned = True
                    break
            if not assigned:
                # Fallback: try standard cost order
                for f in sorted_fac_by_cost[i]:
                    if load[f] + d_i <= capacities[f]:
                        assignment[i] = f
                        load[f] += d_i
                        if load[f] > 0:
                            open_flags[f] = True
                        assigned = True
                        break
        return assignment, load, open_flags

    best_score = float('inf')
    best_solution = None
    
    # Generate initial solutions
    # 1. Standard Greedy (Demand)
    a1, l1, o1 = greedy_init(demands, capacities, assign_cost, sorted_fac_by_cost, mode='demand')
    s1 = calc_cost(a1, l1, o1)
    if s1 < best_score:
        best_score = s1
        best_solution = (a1.copy(), l1.copy(), o1.copy())
        
    # 2. Standard Greedy (Cost)
    a2, l2, o2 = greedy_init(demands, capacities, assign_cost, sorted_fac_by_cost, mode='cost')
    s2 = calc_cost(a2, l2, o2)
    if s2 < best_score:
        best_score = s2
        best_solution = (a2.copy(), l2.copy(), o2.copy())
        
    # 3. Lagrangian Guided
    a3, l3, o3 = lagrangian_guided_init(demands, capacities, assign_cost, sorted_fac_by_cost, open_cost)
    s3 = calc_cost(a3, l3, o3)
    if s3 < best_score:
        best_score = s3
        best_solution = (a3.copy(), l3.copy(), o3.copy())
        
    # Run Local Search on the best initial solution
    # We can also run LS on others if time permits, but let's start with the best.
    # Actually, running LS on all 3 might be better.
    
    solutions = [
        (a1, l1, o1, s1),
        (a2, l2, o2, s2),
        (a3, l3, o3, s3)
    ]
    
    for idx, (a, l, o, s) in enumerate(solutions):
        if time.time() - start_time > time_limit - 5: # Leave 5s for final checks
            break
        
        # Allocate remaining time to this LS run
        time_left = time_limit - (time.time() - start_time)
        # Give it a chunk of time, e.g., 30s per LS run, or split remaining.
        # Let's just run LS until no improvement or time runs out.
        # To be safe, we'll pass a max_time to local_search.
        # If we have 3 solutions, we can't give 100s to each.
        # Let's give 30s to the first, 30s to the second, etc.
        # Or just run the best one for longer.
        
        # Strategy: Run LS on the current best solution for a fixed duration, 
        # then if time remains, try improving other initial solutions.
        
        max_ls_time = min(30.0, time_left)
        
        a_new, l_new, o_new, s_new = local_search(a, l, o, s, max_time=max_ls_time)
        
        if s_new < best_score:
            best_score = s_new
            best_solution = (a_new.copy(), l_new.copy(), o_new.copy())
            
    # If time remains, we can do another pass of LS on the best solution found
    if time.time() - start_time < time_limit - 2:
        if best_solution is not None:
            a_best, l_best, o_best = best_solution
            s_best = best_score
            time_left = time_limit - (time.time() - start_time)
            # Run LS again on the best
            a_new, l_new, o_new, s_new = local_search(a_best, l_best, o_best, s_best, max_time=time_left)
            if s_new < best_score:
                best_score = s_new
                best_solution = (a_new.copy(), l_new.copy(), o_new.copy())

    # Return the assignment
    if best_solution is not None:
        return best_solution[0]
    else:
        # Fallback: return first greedy
        return a1

# EVOLVE-BLOCK-END
