"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    time_limit = 110.0  # Safety margin from 120s
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for efficiency if not already
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    best_solution = None
    best_score = float('inf')
    
    def get_cost(assignment, loads):
        """Calculate total cost for a given assignment and loads."""
        open_facs = np.where(loads > 0)[0]
        open_sum = np.sum(open_cost[open_facs]) if len(open_facs) > 0 else 0.0
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_sum + assign_sum

    def local_search(assignment, loads, open_mask, current_cost):
        """
        Perform 1-opt and 2-opt local search.
        Returns improved assignment, loads, open_mask, cost.
        """
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
                
            # 1-Opt: Move single customer
            for i in range(n_customers):
                if time.time() - start_time > time_limit:
                    break
                current_f = assignment[i]
                current_load_f = loads[current_f]
                
                # Try all other facilities
                # We can vectorize the cost calculation for all target facilities
                # Cost change for moving i from current_f to f:
                # 1. Assign cost change: assign_cost[i, f] - assign_cost[i, current_f]
                # 2. Open cost change:
                #    - If loads[current_f] becomes 0 and it was open: -open_cost[current_f]
                #    - If loads[f] was 0 and it is not open: +open_cost[f]
                
                # Check capacity constraints for all f
                # loads[f] + demands[i] <= capacities[f]
                feasible_facs = np.where(loads + demands[i] <= capacities)[0]
                
                if len(feasible_facs) == 0:
                    continue
                
                # Calculate delta assignment cost
                delta_assign = assign_cost[i, feasible_facs] - assign_cost[i, current_f]
                
                # Calculate delta open cost
                # Case 1: Current facility becomes empty
                becomes_empty = (loads[current_f] == demands[i])
                # Case 2: Target facility becomes non-empty (was empty)
                # Note: feasible_facs includes current_f, but moving to same f is delta 0, so we can ignore or handle
                # Let's exclude current_f from consideration for simplicity, or just let delta be 0
                # Actually, if we move to same f, nothing changes.
                
                # We need to check if target f is currently open (loads[f] > 0)
                # If loads[f] == 0, it is closed. Moving there opens it.
                # If loads[current_f] == demands[i], moving away closes it.
                
                # Vectorized open cost delta:
                # For each f in feasible_facs:
                # delta_open = 0
                # if loads[f] == 0: delta_open += open_cost[f]
                # if loads[current_f] == demands[i] and f != current_f: delta_open -= open_cost[current_f]
                
                # Let's compute this array
                delta_open = np.zeros(len(feasible_facs))
                
                # Identify indices in feasible_facs where load is 0
                empty_targets = np.where(loads[feasible_facs] == 0)[0]
                if len(empty_targets) > 0:
                    delta_open[empty_targets] += open_cost[feasible_facs[empty_targets]]
                    
                # If current facility becomes empty, subtract its open cost from all deltas
                # (except if we are moving to the same facility, which we can filter out)
                if becomes_empty:
                    # Subtract open cost from all, but add back for current_f if it's in feasible_facs
                    delta_open -= open_cost[current_f]
                    # Find index of current_f in feasible_facs
                    idx_current = np.where(feasible_facs == current_f)[0]
                    if len(idx_current) > 0:
                        # If moving to current_f, we didn't actually close it, so add back
                        delta_open[idx_current] += open_cost[current_f]
                
                total_delta = delta_assign + delta_open
                
                # Find the best move (minimum total_delta)
                min_idx = np.argmin(total_delta)
                if total_delta[min_idx] < -1e-9: # Threshold for improvement
                    new_f = feasible_facs[min_idx]
                    
                    # Apply move
                    loads[current_f] -= demands[i]
                    loads[new_f] += demands[i]
                    assignment[i] = new_f
                    current_cost += total_delta[min_idx]
                    improved = True
                    # Update open_mask if needed? We can recalculate or update.
                    # For efficiency, we can just check loads > 0 later.
                    # But for the next iteration's logic, loads are updated.
                    
            # 2-Opt: Swap two customers
            # This is O(N^2 * F). With N=100, this is 10k pairs.
            # For each pair, check if swapping reduces cost.
            # We can limit this to only if 1-opt didn't improve much, or run every few 1-opt passes.
            # Given time constraints, let's try a simplified 2-opt or just rely on 1-opt + restarts.
            # Actually, 2-opt is powerful. Let's implement a basic version.
            
            # To keep it fast, we might skip 2-opt in the tight loop or run it less frequently.
            # Let's try running 2-opt only if no 1-opt improvement was found in the last pass.
            if not improved:
                # Try 2-opt swaps
                for i in range(n_customers):
                    if time.time() - start_time > time_limit:
                        break
                    f_i = assignment[i]
                    for j in range(i + 1, n_customers):
                        if time.time() - start_time > time_limit:
                            break
                        f_j = assignment[j]
                        if f_i == f_j:
                            continue
                        
                        # Check if swapping i to f_j and j to f_i is feasible
                        # loads[f_j] + demands[i] - demands[j] <= capacities[f_j]
                        # loads[f_i] + demands[j] - demands[i] <= capacities[f_i]
                        
                        new_load_fj = loads[f_j] + demands[i] - demands[j]
                        new_load_fi = loads[f_i] + demands[j] - demands[i]
                        
                        if new_load_fj <= capacities[f_j] and new_load_fi <= capacities[f_i]:
                            # Calculate delta cost
                            # Assign cost change:
                            # (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                            delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                           (assign_cost[i, f_i] + assign_cost[j, f_j])
                             
                            # Open cost change:
                            # Check if f_i becomes empty: new_load_fi == 0
                            # Check if f_j becomes empty: new_load_fj == 0
                            # Check if f_i was empty (loads[f_i] == 0) -> becomes non-empty
                            # Check if f_j was empty (loads[f_j] == 0) -> becomes non-empty
                            
                            delta_open = 0.0
                            
                            # f_i status change
                            if loads[f_i] == 0 and new_load_fi > 0:
                                delta_open += open_cost[f_i] # Opened f_i
                            elif loads[f_i] > 0 and new_load_fi == 0:
                                delta_open -= open_cost[f_i] # Closed f_i
                            
                            # f_j status change
                            if loads[f_j] == 0 and new_load_fj > 0:
                                delta_open += open_cost[f_j] # Opened f_j
                            elif loads[f_j] > 0 and new_load_fj == 0:
                                delta_open -= open_cost[f_j] # Closed f_j
                            
                            total_delta = delta_assign + delta_open
                            
                            if total_delta < -1e-9:
                                # Perform swap
                                loads[f_i] = new_load_fi
                                loads[f_j] = new_load_fj
                                assignment[i] = f_j
                                assignment[j] = f_i
                                current_cost += total_delta
                                improved = True
                                # Break out of j loop? Or continue?
                                # Usually, after a move, we restart the scan to avoid inconsistencies
                                # But for speed, we can continue. However, loads changed, so previous j checks are invalid.
                                # Let's break out of the inner loop and restart the outer loop logic?
                                # Simplest: break out of j loop, and the while loop will restart.
                                break
                    if improved:
                        # Restart the 1-opt/2-opt loop from the beginning to ensure consistency
                        # This is safer but slower. Given small N, it's okay.
                        # Actually, just breaking the inner loop and letting the outer `while improved`
                        # restart is fine.
                        pass

        return assignment, loads, current_cost

    # Initialization Strategies
    
    # Strategy 1: Greedy by Demand (Current approach, but improved with local search)
    # Strategy 2: All Open then Close
    
    # Let's run a few random restarts of "All Open then Close" + Local Search
    
    for restart in range(10): # Up to 10 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Initialization: All Open
        # Assign each customer to closest facility
        initial_assignment = np.argmin(assign_cost, axis=1)
        
        # Calculate loads
        loads = np.zeros(n_facilities)
        for i in range(n_customers):
            loads[initial_assignment[i]] += demands[i]
            
        # Check capacity violations. If any, this initialization is invalid.
        # "All Open" might violate capacity if a facility is too small for the customers assigned to it.
        # We need to fix violations first.
        # Simple fix: For any facility with load > capacity, move excess customers to other facilities.
        # This is a feasibility repair step.
        
        feasible = True
        for f in range(n_facilities):
            if loads[f] > capacities[f]:
                feasible = False
                # Move customers from f to other facilities
                # Sort customers assigned to f by "cost to move" (increase in assign cost)
                customers_in_f = np.where(initial_assignment == f)[0]
                # Calculate cost increase for moving each to any other facility
                # For simplicity, move to cheapest feasible other facility
                for i in customers_in_f:
                    # Find best other facility
                    best_f = -1
                    best_cost = float('inf')
                    for g in range(n_facilities):
                        if g == f:
                            continue
                        if loads[g] + demands[i] <= capacities[g]:
                            cost = assign_cost[i, g]
                            if cost < best_cost:
                                best_cost = cost
                                best_f = g
                    if best_f != -1:
                        loads[f] -= demands[i]
                        loads[best_f] += demands[i]
                        initial_assignment[i] = best_f
                    else:
                        # Cannot place customer i anywhere?
                        # This shouldn't happen if total capacity >= total demand
                        # If it does, we have a problem. Assume problem is feasible.
                        pass
                # Re-check load
                if loads[f] > capacities[f]:
                    # Still infeasible?
                    pass
                    
        # Now, "Close if profitable" phase
        # Iterate until no more closures
        changed = True
        while changed:
            changed = False
            for f in range(n_facilities):
                if loads[f] == 0:
                    continue # Already closed
                # Calculate cost to keep f open
                # Cost_keep = open_cost[f] + sum(assign_cost[i, f] for i in f)
                # Cost_close = sum(assign_cost[i, best_g(i)) for i in f)
                # where best_g(i) is cheapest other open facility with capacity
                
                # Get customers in f
                custs = np.where(initial_assignment == f)[0]
                if len(custs) == 0:
                    continue
                    
                cost_keep_assign = np.sum(assign_cost[custs, f])
                
                cost_close_assign = 0.0
                feasible_close = True
                # Simulate moving all customers from f to others
                # We need to track temporary loads
                temp_loads = loads.copy()
                temp_loads[f] = 0 # f is closed
                
                for i in custs:
                    # Find best g != f with capacity
                    best_g = -1
                    best_c = float('inf')
                    for g in range(n_facilities):
                        if g == f:
                            continue
                        if temp_loads[g] + demands[i] <= capacities[g]:
                            c = assign_cost[i, g]
                            if c < best_c:
                                best_c = c
                                best_g = g
                    if best_g == -1:
                        feasible_close = False
                        break
                    cost_close_assign += best_c
                    temp_loads[best_g] += demands[i]
                    
                if feasible_close:
                    delta = cost_close_assign - cost_keep_assign - open_cost[f]
                    if delta < -1e-9:
                        # Close f
                        # Update assignment and loads
                        # We need to actually perform the moves
                        # Re-run the assignment for these customers
                        # Note: The "best_g" might change as we move customers one by one
                        # So we should do it sequentially
                        for i in custs:
                            # Find best g != f with capacity in CURRENT loads
                            # But we are closing f, so loads[f] is decreasing
                            # Let's update loads[f] as we go
                            best_g = -1
                            best_c = float('inf')
                            for g in range(n_facilities):
                                if g == f:
                                    continue
                                if loads[g] + demands[i] <= capacities[g]:
                                    c = assign_cost[i, g]
                                    if c < best_c:
                                        best_c = c
                                        best_g = g
                            if best_g != -1:
                                loads[f] -= demands[i]
                                loads[best_g] += demands[i]
                                initial_assignment[i] = best_g
                            else:
                                # Infeasible? Should not happen if feasible_close was true
                                pass
                        changed = True
                        # Restart the while loop to check other facilities
                        break
                        
        # Now run local search on this solution
        current_cost = get_cost(initial_assignment, loads)
        assignment_ls, loads_ls, cost_ls = local_search(initial_assignment.copy(), loads.copy(), None, current_cost)
        
        if cost_ls < best_score:
            best_score = cost_ls
            best_solution = assignment_ls.copy()
            
    if best_solution is None:
        # Fallback to simple greedy if something went wrong
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        best_solution = result

    return best_solution
# EVOLVE-BLOCK-END
