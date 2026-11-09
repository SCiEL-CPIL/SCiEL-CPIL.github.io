"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # 1. Greedy Initialization
    # Sort customers by demand descending to place large items first
    order = np.argsort(-demands)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    is_open = np.zeros(n_facilities, dtype=bool)
    
    for i in order:
        # Find feasible facilities
        feasible = load + demands[i] <= capacities
        if not np.any(feasible):
            # Should not happen if total demand <= total capacity, but just in case
            # Assign to least full facility
            f = np.argmin(load)
        else:
            # Among feasible, pick cheapest
            costs = assign_cost[i].copy()
            costs[~feasible] = np.inf
            f = np.argmin(costs)
        
        assignment[i] = f
        load[f] += demands[i]
        is_open[f] = True

    def calculate_cost(assignment, load, is_open):
        # Sum of assignment costs
        cost = np.sum(assign_cost[np.arange(n_customers), assignment])
        # Sum of open costs
        cost += np.sum(open_cost[is_open])
        return cost

    best_cost = calculate_cost(assignment, load, is_open)
    best_assignment = assignment.copy()
    best_load = load.copy()
    best_is_open = is_open.copy()

    no_improvement_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Pass 1: Customer Reassignment
        # We iterate customers in random order to break symmetry
        for i in np.random.permutation(n_customers):
            if time.time() - start_time > time_limit:
                break
            
            f_curr = assignment[i]
            if not is_open[f_curr]:
                # This shouldn't happen in a valid state
                continue
                
            # Calculate potential gains for moving to all other facilities
            # Delta assignment cost
            delta_assign = assign_cost[i] - assign_cost[i, f_curr]
            
            # Delta open cost
            # If f_curr becomes empty, we save open_cost[f_curr]
            becomes_empty = (load[f_curr] == demands[i])
            # If f_new was empty, we pay open_cost[f_new]
            was_empty = (~is_open)
            
            # We can only move to feasible facilities
            feasible = load + demands[i] <= capacities
            
            # Effective delta
            # Base: delta_assign
            # If becomes_empty, subtract open_cost[f_curr] for ALL moves? 
            # No, the saving is realized regardless of where we move to, 
            # BUT we must ensure we don't double count or misapply.
            # Actually, the open cost of f_curr is saved ONLY IF we move the last customer out.
            # So, if becomes_empty, delta_open = -open_cost[f_curr] for any valid move.
            # If target f_new is currently closed, delta_open += open_cost[f_new].
            
            delta_open = np.zeros(n_facilities)
            if becomes_empty:
                delta_open[:] = -open_cost[f_curr]
            
            # Add opening cost for closed facilities
            delta_open[was_empty] += open_cost[was_empty]
            
            total_delta = delta_assign + delta_open
            
            # Mask infeasible
            total_delta[~feasible] = np.inf
            total_delta[f_curr] = np.inf # Don't move to same
            
            best_f = np.argmin(total_delta)
            min_delta = total_delta[best_f]
            
            if min_delta < -1e-9:
                # Perform move
                # Update load
                load[f_curr] -= demands[i]
                if load[f_curr] < 1e-9:
                    load[f_curr] = 0
                    is_open[f_curr] = False
                
                load[best_f] += demands[i]
                if not is_open[best_f]:
                    is_open[best_f] = True
                
                assignment[i] = best_f
                improved = True
        
        # Pass 2: Facility Opening (try to open closed facilities)
        # For each closed facility, see if opening it allows significant savings
        closed_facilities = np.where(~is_open)[0]
        for f_new in closed_facilities:
            if time.time() - start_time > time_limit:
                break
            
            # Which customers would benefit from moving to f_new?
            # Gain = assign_cost[i, assignment[i]] - assign_cost[i, f_new]
            gains = assign_cost[np.arange(n_customers), assignment] - assign_cost[:, f_new]
            
            # Only consider customers that can fit in f_new
            # Capacity of f_new is capacities[f_new]
            # We want to pick a subset of customers with highest gains such that sum(demands) <= capacities[f_new]
            # And sum(gains) > open_cost[f_new]
            
            # Sort by gain descending
            # Filter out negative gains? Yes, we only want positive gains.
            positive_gain_idx = np.where(gains > 0)[0]
            if len(positive_gain_idx) == 0:
                continue
            
            # Sort these indices by gain
            sorted_idx = positive_gain_idx[np.argsort(-gains[positive_gain_idx])]
            
            total_gain = 0
            total_demand = 0
            selected = []
            
            for idx in sorted_idx:
                if total_demand + demands[idx] > capacities[f_new]:
                    continue
                total_gain += gains[idx]
                total_demand += demands[idx]
                selected.append(idx)
                
                # Early stop if gain is already high? No, we need max gain within capacity.
                # But if gain stops being positive, we stop.
                # Since we sorted by gain, once gain <= 0, we stop.
                # The loop above iterates sorted_idx which are all > 0 gains.
            
            if total_gain > open_cost[f_new] + 1e-9:
                # Open f_new and move selected customers
                is_open[f_new] = True
                load[f_new] = total_demand
                
                for idx in selected:
                    f_old = assignment[idx]
                    load[f_old] -= demands[idx]
                    if load[f_old] < 1e-9:
                        load[f_old] = 0
                        is_open[f_old] = False
                    assignment[idx] = f_new
                
                improved = True

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= 5:
                break
        else:
            no_improvement_count = 0
            
        # Update best
        current_cost = calculate_cost(assignment, load, is_open)
        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = assignment.copy()
            best_load = load.copy()
            best_is_open = is_open.copy()

    return best_assignment
# EVOLVE-BLOCK-END
