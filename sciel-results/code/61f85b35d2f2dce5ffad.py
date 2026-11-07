"""Improved program using Local Search with Greedy Reassignment for Capacitated Facility Location."""
# EVOLVE-BLOCK-START
import numpy as np

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a Local Search heuristic.
    
    1. Initial solution: Greedy assignment with all facilities open.
    2. Improvement steps:
       a. Move: Move a single customer to a different facility if it reduces cost.
       b. Swap: Swap two customers between facilities if it reduces cost.
       c. Close: Close a facility and reassign its customers to others if it reduces total cost.
       d. Open: Open a closed facility and assign customers to it if it reduces total cost.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    if n_customers == 0:
        return np.array([], dtype=int)

    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)

    # State
    assignment = np.zeros(n_customers, dtype=int)
    open_facilities = np.ones(n_facilities, dtype=bool)
    load = np.zeros(n_facilities, dtype=int)
    
    # Helper to calculate total cost
    def calc_total_cost():
        cost = np.sum(assign_cost[np.arange(n_customers), assignment])
        cost += np.sum(open_cost[open_facilities])
        return cost

    # 1. Initial Greedy Assignment
    # Sort customers by demand descending
    order = np.argsort(-demands)
    for i in order:
        # Find feasible open facilities
        feasible = [f for f in range(n_facilities) if open_facilities[f] and load[f] + demands[i] <= capacities[f]]
        if feasible:
            # Pick cheapest
            best_f = min(feasible, key=lambda f: assign_cost[i, f])
            assignment[i] = best_f
            load[best_f] += demands[i]
        else:
            # If no feasible facility (shouldn't happen if sum caps >= sum demands and all open initially)
            # Force assign to one with most space, even if over capacity? 
            # Problem states capacities are integers, we must not exceed.
            # If we can't assign, the instance might be infeasible? 
            # "capacities, integers per facility"
            # Assuming feasible instances.
            # Fallback: pick facility with max remaining capacity
            remaining = capacities - load
            best_f = np.argmax(remaining)
            assignment[i] = best_f
            load[best_f] += demands[i]

    current_cost = calc_total_cost()
    
    # 2. Local Search
    max_iters = 500
    for _ in range(max_iters):
        improved = False
        
        # --- Phase 1: Move ---
        # For each customer, try moving to another facility
        # We can do this in random order or fixed order. Fixed is fine for deterministic.
        for i in range(n_customers):
            old_f = assignment[i]
            if not open_facilities[old_f]:
                # This shouldn't happen if we maintain consistency
                continue
                
            best_gain = 0
            best_new_f = -1
            
            for new_f in range(n_facilities):
                if new_f == old_f:
                    continue
                if not open_facilities[new_f]:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Cost change: new_assign - old_assign
                # Fixed cost doesn't change
                gain = assign_cost[i, old_f] - assign_cost[i, new_f]
                if gain > best_gain:
                    best_gain = gain
                    best_new_f = new_f
            
            if best_new_f != -1:
                # Perform move
                load[old_f] -= demands[i]
                load[best_new_f] += demands[i]
                assignment[i] = best_new_f
                current_cost -= best_gain
                improved = True
        
        # --- Phase 2: Swap ---
        # For each pair of customers, try swapping
        # To keep it fast, we can limit checks or just do all. 100*100/2 = 5000 pairs.
        # Each swap check is O(1).
        for i in range(n_customers):
            for k in range(i + 1, n_customers):
                f_i = assignment[i]
                f_k = assignment[k]
                if f_i == f_k:
                    continue
                if not open_facilities[f_i] or not open_facilities[f_k]:
                    continue
                
                # Check capacity for swap
                # i moves to f_k, k moves to f_i
                # load[f_i] - d_i + d_k <= cap[f_i]
                # load[f_k] - d_k + d_i <= cap[f_k]
                if load[f_i] - demands[i] + demands[k] > capacities[f_i]:
                    continue
                if load[f_k] - demands[k] + demands[i] > capacities[f_k]:
                    continue
                
                # Cost change
                old_cost = assign_cost[i, f_i] + assign_cost[k, f_k]
                new_cost = assign_cost[i, f_k] + assign_cost[k, f_i]
                gain = old_cost - new_cost
                
                if gain > 0:
                    # Perform swap
                    load[f_i] = load[f_i] - demands[i] + demands[k]
                    load[f_k] = load[f_k] - demands[k] + demands[i]
                    assignment[i], assignment[k] = f_k, f_i
                    current_cost -= gain
                    improved = True
        
        # --- Phase 3: Close Facilities ---
        # Try closing each open facility
        for f in range(n_facilities):
            if not open_facilities[f]:
                continue
            
            # Identify customers in f
            customers_in_f = np.where(assignment == f)[0]
            if len(customers_in_f) == 0:
                # Close it for free
                open_facilities[f] = False
                current_cost -= open_cost[f]
                improved = True
                continue
            
            # Simulate closing f
            # Reassign customers_in_f to other open facilities
            # Use Greedy: sort by demand desc
            temp_assignment = assignment.copy()
            temp_load = load.copy()
            temp_load[f] = 0 # Clear load for f
            temp_open = open_facilities.copy()
            temp_open[f] = False
            
            # Save original loads of affected customers to restore if we revert? 
            # No, we just need to check if the new solution is better.
            # We need to reassign customers_in_f.
            
            # Sort customers by demand descending for greedy reassignment
            cust_order = customers_in_f[np.argsort(-demands[customers_in_f])]
            
            feasible = True
            new_assign_cost = 0
            # Calculate cost for customers NOT in f (they stay same)
            # Cost for customers IN f will be recalculated
            
            # We need to know the cost of the reassignment
            # Let's just do the reassignment on a copy of the state
            
            # To be efficient, we only reassign customers_in_f.
            # Other customers remain assigned to their current facilities.
            
            # Check if reassignment is possible
            for i in cust_order:
                # Find best feasible facility among temp_open
                best_f_new = -1
                best_cost = np.inf
                for nf in range(n_facilities):
                    if not temp_open[nf]:
                        continue
                    if temp_load[nf] + demands[i] <= capacities[nf]:
                        if assign_cost[i, nf] < best_cost:
                            best_cost = assign_cost[i, nf]
                            best_f_new = nf
                
                if best_f_new == -1:
                    feasible = False
                    break
                else:
                    temp_load[best_f_new] += demands[i]
                    temp_assignment[i] = best_f_new
                    new_assign_cost += best_cost
            
            if not feasible:
                continue
            
            # Calculate total new cost
            # Assign cost for customers not in f
            mask_not_f = assignment != f
            cost_others = np.sum(assign_cost[mask_not_f, assignment[mask_not_f]])
            
            # Wait, this is expensive to calculate every time.
            # We can track current_assign_cost separately.
            
            # Let's optimize:
            # current_assign_cost = sum(assign_cost[i, assignment[i]])
            # cost_removed = sum(assign_cost[i, f] for i in customers_in_f)
            # new_cost_added = new_assign_cost
            # delta_assign = new_assign_cost - cost_removed
            # delta_fixed = -open_cost[f]
            # total_delta = delta_assign + delta_fixed
            
            cost_removed = np.sum(assign_cost[customers_in_f, f])
            delta_assign = new_assign_cost - cost_removed
            delta_fixed = -open_cost[f]
            
            if delta_assign + delta_fixed < 0:
                # Commit
                # Update loads
                # Load of f becomes 0
                # Load of others updated via temp_load
                # We need to update the global load array
                # temp_load contains the new loads for all facilities?
                # No, temp_load was copied from load.
                # We only updated temp_load for the reassignment.
                # But we also need to ensure that the loads for facilities NOT in f are correct.
                # They were correct in temp_load (copied from load).
                # We added the new demands to temp_load.
                # So temp_load is the correct new load vector.
                
                load = temp_load
                assignment = temp_assignment
                open_facilities[f] = False
                current_cost += delta_assign + delta_fixed
                improved = True
        
        # --- Phase 4: Open Facilities ---
        # Try opening each closed facility
        for f in range(n_facilities):
            if open_facilities[f]:
                continue
            
            # Simulate opening f
            # We want to move customers to f to reduce assignment cost.
            # Greedy: Sort customers by savings (current_cost - cost_to_f) desc.
            
            # Calculate potential savings for each customer
            # current_f = assignment[i]
            # saving = assign_cost[i, current_f] - assign_cost[i, f]
            
            savings = np.zeros(n_customers)
            for i in range(n_customers):
                cur_f = assignment[i]
                savings[i] = assign_cost[i, cur_f] - assign_cost[i, f]
            
            # Sort indices by savings desc
            # Only consider customers with positive savings
            positive_idx = np.where(savings > 0)[0]
            if len(positive_idx) == 0:
                continue
                
            pos_order = positive_idx[np.argsort(-savings[positive_idx])]
            
            # Try to assign customers to f
            temp_load = load.copy()
            temp_assignment = assignment.copy()
            temp_load[f] = 0 # Start empty
            
            total_savings = 0
            assigned_count = 0
            
            for i in pos_order:
                if temp_load[f] + demands[i] <= capacities[f]:
                    # Move i to f
                    old_f = temp_assignment[i]
                    # Update load
                    temp_load[old_f] -= demands[i]
                    temp_load[f] += demands[i]
                    temp_assignment[i] = f
                    total_savings += savings[i]
                    assigned_count += 1
                else:
                    # If we can't fit this customer, can we fit a smaller one later?
                    # Since we sorted by savings, not demand, we might miss a combination.
                    # But for a heuristic, this is acceptable.
                    # We could try to swap? No, keep it simple.
                    pass
            
            # Check if total savings > open cost
            if total_savings > open_cost[f]:
                # Commit
                load = temp_load
                assignment = temp_assignment
                open_facilities[f] = True
                current_cost -= total_savings + open_cost[f]
                improved = True
        
        if not improved:
            break

    return assignment
# EVOLVE-BLOCK-END
