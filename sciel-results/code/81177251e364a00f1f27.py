"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays if needed
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Initial solution: greedy assignment
    result = np.full(n_customers, -1, dtype=np.int64)
    load = np.zeros(n_facilities, dtype=np.int64)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    for i in order:
        # Find feasible facilities
        feasible = np.where(capacities - load >= demands[i])[0]
        if len(feasible) == 0:
            # Fallback: pick facility with max remaining capacity
            # This might lead to infeasibility if total demand > total capacity,
            # but we assume a feasible solution exists.
            best_f = np.argmax(capacities - load)
        else:
            # Pick the cheapest feasible facility
            costs = assign_cost[i, feasible]
            best_f = feasible[np.argmin(costs)]
        
        result[i] = best_f
        load[best_f] += demands[i]

    # Repair function to fix any capacity violations
    def repair():
        nonlocal result, load
        for _ in range(100):
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
            
            # Pick the most overloaded facility
            f_over = overloaded[np.argmax(load[overloaded] - capacities[overloaded])]
            
            # Find customers in f_over
            customers_in_f = [i for i in range(n_customers) if result[i] == f_over]
            if not customers_in_f:
                continue
                
            # Sort customers by demand ascending (easiest to fit elsewhere)
            customers_in_f.sort(key=lambda x: demands[x])
            
            # Sort facilities by remaining capacity descending
            remaining_cap = capacities - load
            f2_candidates = np.argsort(-remaining_cap)
            
            moved = False
            for i in customers_in_f:
                for f2_idx in f2_candidates:
                    f2 = f2_idx
                    if f2 == f_over:
                        continue
                    if load[f2] + demands[i] <= capacities[f2]:
                        # Move i to f2
                        load[f_over] -= demands[i]
                        load[f2] += demands[i]
                        result[i] = f2
                        moved = True
                        break
                if moved:
                    break
            
            if not moved:
                break

    # Run repair after initialization
    repair()

    def compute_total_cost(assignment, loads):
        total = 0.0
        used = set(assignment.tolist())
        for f in used:
            total += open_cost[f]
        for i in range(n_customers):
            total += assign_cost[i, assignment[i]]
        return total
    
    current_cost = compute_total_cost(result, load)
    
    def try_improve(assignment, loads):
        best_cost = current_cost
        best_assignment = assignment.copy()
        best_loads = loads.copy()
        
        # 1-Move
        for i in range(n_customers):
            cur_f = assignment[i]
            for f in range(n_facilities):
                if f == cur_f:
                    continue
                if loads[f] + demands[i] > capacities[f]:
                    continue
                
                new_loads = loads.copy()
                new_loads[cur_f] -= demands[i]
                new_loads[f] += demands[i]
                
                cost_change = assign_cost[i, f] - assign_cost[i, cur_f]
                
                cur_f_empty_after = (new_loads[cur_f] == 0)
                f_empty_before = (loads[f] == 0)
                
                if cur_f_empty_after:
                    cost_change -= open_cost[cur_f]
                if f_empty_before:
                    cost_change += open_cost[f]
                
                new_cost = best_cost + cost_change
                if new_cost < best_cost - 1e-9:
                    best_cost = new_cost
                    best_assignment = assignment.copy()
                    best_assignment[i] = f
                    best_loads = new_loads
        
        # 2-Swap
        for f1 in range(n_facilities):
            customers_f1 = [i for i in range(n_customers) if assignment[i] == f1]
            if not customers_f1:
                continue
            for f2 in range(f1 + 1, n_facilities):
                customers_f2 = [i for i in range(n_customers) if assignment[i] == f2]
                if not customers_f2:
                    continue
                
                for i in customers_f1:
                    for j in customers_f2:
                        new_load_f1 = loads[f1] - demands[i] + demands[j]
                        new_load_f2 = loads[f2] - demands[j] + demands[i]
                        
                        if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                            continue
                        
                        cost_change = (assign_cost[i, f2] - assign_cost[i, f1]) + \
                                      (assign_cost[j, f1] - assign_cost[j, f2])
                        
                        f1_empty_after = (new_load_f1 == 0)
                        f2_empty_after = (new_load_f2 == 0)
                        f1_empty_before = (loads[f1] == 0)
                        f2_empty_before = (loads[f2] == 0)
                        
                        if f1_empty_after and not f1_empty_before:
                            cost_change -= open_cost[f1]
                        elif not f1_empty_after and f1_empty_before:
                            cost_change += open_cost[f1]
                        
                        if f2_empty_after and not f2_empty_before:
                            cost_change -= open_cost[f2]
                        elif not f2_empty_after and f2_empty_before:
                            cost_change += open_cost[f2]
                        
                        new_cost = best_cost + cost_change
                        if new_cost < best_cost - 1e-9:
                            best_cost = new_cost
                            best_assignment = assignment.copy()
                            best_assignment[i] = f2
                            best_assignment[j] = f1
                            best_loads = loads.copy()
                            best_loads[f1] = new_load_f1
                            best_loads[f2] = new_load_f2
        
        return best_assignment, best_loads, best_cost
    
    improved = True
    max_iters = 50
    iters = 0
    while improved and iters < max_iters:
        if time.time() > deadline:
            break
        improved = False
        new_assignment, new_loads, new_cost = try_improve(result, load)
        if new_cost < current_cost - 1e-9:
            result = new_assignment
            load = new_loads
            current_cost = new_cost
            improved = True
        iters += 1
    
    # Final repair and check
    repair()
    
    # Verify indices
    for i in range(n_customers):
        if result[i] < 0 or result[i] >= n_facilities:
             for f in range(n_facilities):
                 if load[f] + demands[i] <= capacities[f]:
                     old_f = result[i]
                     if old_f >= 0 and old_f < n_facilities:
                         load[old_f] -= demands[i]
                     result[i] = f
                     load[f] += demands[i]
                     break
             
    return result
# EVOLVE-BLOCK-END
