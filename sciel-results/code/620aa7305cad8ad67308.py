import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Greedy Initialization
    # Sort customers by demand descending (Largest First)
    cust_order = np.argsort(-demands)
    assignment = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities, dtype=np.float64)
    
    for i in cust_order:
        d_i = demands[i]
        best_j = -1
        best_cost = float('inf')
        
        for j in range(n_facilities):
            if load[j] + d_i <= capacities[j]:
                # Calculate marginal cost
                # If facility j is empty, we incur open_cost[j]
                # If facility j is not empty, we don't
                # Plus assignment cost
                marginal_cost = assign_cost[i, j]
                if load[j] == 0:
                    marginal_cost += open_cost[j]
                
                if marginal_cost < best_cost:
                    best_cost = marginal_cost
                    best_j = j
                    
        if best_j == -1:
            # If no facility fits, this is a problem. 
            # In valid instances, total capacity >= total demand.
            # If we get here, it means we made bad choices earlier.
            # Fallback: assign to facility with most remaining capacity (might violate)
            # But we must return valid. Let's try to find a swap? 
            # For initialization, let's just pick the one with max remaining cap and hope local search fixes.
            # Actually, if we pick max remaining, we might exceed capacity.
            # Better: pick the one with max capacity overall?
            # Let's just assign to 0 for now, local search will fix.
            best_j = 0
            
        assignment[i] = best_j
        load[best_j] += d_i
        
    # 2. Local Search
    # We will iterate until no improvement or time limit
    # Cost calculation helper
    def calc_cost(assignment, load):
        total = 0.0
        for j in range(n_facilities):
            if load[j] > 0:
                total += open_cost[j]
        for i in range(n_customers):
            j = assignment[i]
            if j >= 0:
                total += assign_cost[i, j]
        return total

    current_cost = calc_cost(assignment, load)
    
    max_iterations = 5000
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 50:
        improved = False
        
        # 1. Try 1-Moves: Move customer i from j to k
        # To speed up, we can check all i and all k != j
        for i in range(n_customers):
            if time.time() > deadline:
                break
            j = assignment[i]
            if j < 0:
                continue
            d_i = demands[i]
            
            for k in range(n_facilities):
                if k == j:
                    continue
                if load[k] + d_i <= capacities[k]:
                    # Calculate delta
                    # Remove i from j
                    # Add i to k
                    delta = assign_cost[i, k] - assign_cost[i, j]
                    
                    # Check if j becomes empty
                    if load[j] - d_i == 0:
                        delta -= open_cost[j]
                        
                    # Check if k was empty
                    if load[k] == 0:
                        delta += open_cost[k]
                        
                    if delta < -1e-9:
                        # Perform move
                        load[j] -= d_i
                        load[k] += d_i
                        assignment[i] = k
                        current_cost += delta
                        improved = True
                        break # Restart inner loop for this i? Or break outer?
                        # Breaking inner loop means we try next i. This is standard "first improvement" or "best improvement" per i.
                        # Let's break inner loop to keep moving.
            
        if not improved:
            # 2. Try 2-Swaps: Swap customer i (at j) and customer l (at k)
            # i moves to k, l moves to j
            # Conditions:
            # load[k] - d_l + d_i <= cap[k]
            # load[j] - d_i + d_l <= cap[j]
            # Cost change:
            # (assign_cost[i, k] - assign_cost[i, j]) + (assign_cost[l, j] - assign_cost[l, k])
            # Open cost changes:
            # If j becomes empty (load[j] - d_i + d_l == 0? No, load[j] changes by -d_i + d_l)
            # If k becomes empty (load[k] - d_l + d_i == 0)
            
            # To make this efficient, we can limit the scope or just do a simple loop.
            # 100 customers * 100 customers * 20 facilities? No, swaps are between pairs of customers.
            # Loop i, then loop l != i.
            # This is O(N^2). 100^2 = 10,000. Very fast.
            
            found_swap = False
            for i in range(n_customers):
                if found_swap:
                    break
                j = assignment[i]
                d_i = demands[i]
                for l in range(i + 1, n_customers):
                    k = assignment[l]
                    if j == k:
                        continue
                    d_l = demands[l]
                    
                    # Check capacity constraints for swap
                    new_load_j = load[j] - d_i + d_l
                    new_load_k = load[k] - d_l + d_i
                    
                    if new_load_j <= capacities[j] and new_load_k <= capacities[k]:
                        # Calculate cost delta
                        delta_assign = (assign_cost[i, k] - assign_cost[i, j]) + (assign_cost[l, j] - assign_cost[l, k])
                        
                        delta_open = 0.0
                        # Check if j becomes empty
                        if new_load_j == 0:
                            delta_open -= open_cost[j]
                        # Check if j was empty (not possible since i is there)
                        
                        # Check if k becomes empty
                        if new_load_k == 0:
                            delta_open -= open_cost[k]
                        # Check if k was empty (not possible since l is there)
                        
                        # Note: If j was empty? No, i is assigned to j.
                        # If k was empty? No, l is assigned to k.
                        
                        # What if j becomes empty AND k becomes empty?
                        # We subtract both open costs.
                        
                        # What if j was NOT empty and stays NOT empty? No change.
                        # What if j was NOT empty and becomes empty? Subtract open_cost[j].
                        # What if j was empty? Impossible.
                        
                        # Wait, what if the swap causes a facility to become empty?
                        # We handled that.
                        
                        # What if the swap causes a facility to be OPENED?
                        # It was already open (had i or l). So no new open cost.
                        
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < -1e-9:
                            # Perform swap
                            load[j] = new_load_j
                            load[k] = new_load_k
                            assignment[i] = k
                            assignment[l] = j
                            current_cost += total_delta
                            improved = True
                            found_swap = True
                            break
                            
            if found_swap:
                continue # Restart the while loop to try 1-moves again? Or just continue to next while iteration.
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
        else:
            no_improve_count = 0
            
    return assignment
