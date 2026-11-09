import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Solves the Capacitated Facility Location Problem using a heuristic 
    initialization followed by local search optimization.
    
    Args:
    demands: array of customer demands
    capacities: array of facility capacities
    open_cost: array of facility opening costs
    assign_cost: matrix of assignment costs
    
    Returns:
    numpy array of facility indices for each customer
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Greedy Initialization: Largest Demand First
    # Sort customers by demand descending
    order = np.argsort(-demands)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    # Pre-sort facility indices by cost for each customer to speed up greedy?
    # For 100 customers, just sorting on the fly is fine.
    for i in order:
        d = demands[i]
        # Find feasible facility with minimum (assign_cost + open_cost if empty)
        # We just use assign_cost for greedy simplicity, or include open cost penalty?
        # Including open cost penalty helps avoid opening many small facilities.
        # Let's use assign_cost[i, f] + (open_cost[f] if load[f] == 0 else 0)
        
        best_f = -1
        best_cost = np.inf
        for f in range(n_facilities):
            if load[f] + d <= capacities[f]:
                c = assign_cost[i, f]
                if load[f] == 0:
                    c += open_cost[f]
                if c < best_cost:
                    best_cost = c
                    best_f = f
        
        if best_f == -1:
            # Should not happen if total capacity is sufficient, but fallback
            # Assign to any feasible facility with min assign cost
            for f in range(n_facilities):
                if load[f] + d <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                # No feasible facility, return zeros (invalid but avoids crash)
                return np.zeros(n_customers, dtype=int)
                
        assignment[i] = best_f
        load[best_f] += d

    # Helper to calculate total cost
    def calc_cost(assignment):
        total = 0.0
        used_facilities = set(assignment)
        for f in used_facilities:
            total += open_cost[f]
        for i in range(n_customers):
            total += assign_cost[i, assignment[i]]
        return total

    current_cost = calc_cost(assignment)
    
    # 2. Local Search
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # Pass 1: Single Moves (Best Improvement)
        best_move_cost = current_cost
        best_move_i = -1
        best_move_f_new = -1
        best_move_f_old = -1
        
        for i in range(n_customers):
            f_old = assignment[i]
            d = demands[i]
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                # Check capacity
                if load[f_new] + d > capacities[f_new]:
                    continue
                
                # Calculate cost change
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                
                # Open cost change
                delta_open = 0.0
                # If f_old becomes empty, we save open_cost[f_old]
                if load[f_old] == d:
                    delta_open -= open_cost[f_old]
                # If f_new was empty, we pay open_cost[f_new]
                if load[f_new] == 0:
                    delta_open += open_cost[f_new]
                
                new_cost = current_cost + delta_assign + delta_open
                
                if new_cost < best_move_cost:
                    best_move_cost = new_cost
                    best_move_i = i
                    best_move_f_new = f_new
                    best_move_f_old = f_old
        
        if best_move_i != -1:
            # Apply move
            i = best_move_i
            f_old = best_move_f_old
            f_new = best_move_f_new
            d = demands[i]
            
            load[f_old] -= d
            load[f_new] += d
            assignment[i] = f_new
            current_cost = best_move_cost
            improved = True
            continue # Restart search from this new state
        
        # Pass 2: Swaps (First Improvement)
        # Swap customer i at f1 with customer j at f2 (f1 != f2)
        # We iterate i, j, and check if swapping helps.
        # To make this efficient, we can iterate pairs of customers.
        # N=100, so 4950 pairs. For each pair, check if they are in different facilities.
        # If so, check if swap is valid (capacities are preserved, so always valid for loads).
        # Cost change: (c[i, f2] + c[j, f1]) - (c[i, f1] + c[j, f2])
        # Open cost change: 0 unless a facility becomes empty (unlikely in swap if loads > 0, 
        # but possible if load was exactly d_i and we move d_i out, and d_j > 0 moves in? 
        # No, if we swap i from f1 to f2 and j from f2 to f1:
        # load[f1] changes by -d_i + d_j. If load[f1] was d_i, it becomes d_j. If d_j > 0, not empty.
        # If load[f1] was d_i and d_j == 0? Demands are >= 5. So load never becomes 0 if it was > 0 
        # unless d_i == load[f1] and d_j == 0, which is impossible.
        # So open costs do not change in swaps.
        
        for i in range(n_customers):
            f1 = assignment[i]
            d_i = demands[i]
            for j in range(i + 1, n_customers):
                f2 = assignment[j]
                if f1 == f2:
                    continue
                
                d_j = demands[j]
                
                # Capacity check for swap:
                # f1 receives d_j, loses d_i. New load: load[f1] - d_i + d_j <= cap[f1]
                # f2 receives d_i, loses d_j. New load: load[f2] - d_j + d_i <= cap[f2]
                if load[f1] - d_i + d_j > capacities[f1]:
                    continue
                if load[f2] - d_j + d_i > capacities[f2]:
                    continue
                
                # Cost change
                cost_old = assign_cost[i, f1] + assign_cost[j, f2]
                cost_new = assign_cost[i, f2] + assign_cost[j, f1]
                
                if cost_new < cost_old:
                    # Apply swap
                    load[f1] = load[f1] - d_i + d_j
                    load[f2] = load[f2] - d_j + d_i
                    assignment[i] = f2
                    assignment[j] = f1
                    current_cost = current_cost - (cost_old - cost_new)
                    improved = True
                    break # Restart search after first improvement
            if improved:
                break
        
        if not improved:
            break
            
    return assignment
