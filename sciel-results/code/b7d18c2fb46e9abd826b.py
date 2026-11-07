# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for overhead

    # Phase 1: Greedy initialization
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    count = np.zeros(n_facilities, dtype=int)
    
    for i in order:
        best_f = -1
        best_score = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] > capacities[f]:
                continue
            if count[f] == 0:
                score = assign_cost[i, f] + open_cost[f]
            else:
                score = assign_cost[i, f] + open_cost[f] / (count[f] + 1)
            if score < best_score:
                best_score = score
                best_f = f
        if best_f == -1:
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    best_f = f
                    break
            if best_f == -1:
                best_f = np.argmax(capacities - load)
        result[i] = best_f
        load[best_f] += demands[i]
        count[best_f] += 1
        
    # Calculate initial cost
    open_mask = (count > 0)
    total_cost = np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_customers), result])
    
    best_result = result.copy()
    best_total_cost = total_cost

    def local_search(result, load, count, total_cost):
        improved = True
        while improved:
            improved = False
            # Pass 1: Relocations
            for i in range(n_customers):
                old_f = result[i]
                best_new_f = -1
                best_delta = 0.0
                
                for f in range(n_facilities):
                    if f == old_f:
                        continue
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    
                    # Delta assignment cost
                    d_assign = assign_cost[i, f] - assign_cost[i, old_f]
                    
                    # Delta open cost
                    d_open = 0.0
                    if count[old_f] == 1:
                        d_open -= open_cost[old_f]
                    if count[f] == 0:
                        d_open += open_cost[f]
                    
                    delta = d_assign + d_open
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = f
                
                if best_new_f != -1:
                    result[i] = best_new_f
                    load[old_f] -= demands[i]
                    load[best_new_f] += demands[i]
                    count[old_f] -= 1
                    count[best_new_f] += 1
                    total_cost += best_delta
                    improved = True
            
            # Pass 2: Swaps
            for i in range(n_customers):
                for j in range(i + 1, n_customers):
                    f_i = result[i]
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    
                    # Check capacities for swap
                    if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                        continue
                    if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                        continue
                    
                    # Cost change (Open costs invariant for swaps between distinct facilities)
                    d_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    if d_assign < -1e-9:
                        # Perform swap
                        result[i], result[j] = result[j], result[i]
                        load[f_i] = load[f_i] - demands[i] + demands[j]
                        load[f_j] = load[f_j] - demands[j] + demands[i]
                        # counts unchanged
                        total_cost += d_assign
                        improved = True
                        
        return total_cost

    # ILS Loop
    while time.time() - start_time < time_limit:
        # Local search to convergence
        total_cost = local_search(result, load, count, total_cost)
        
        if total_cost < best_total_cost - 1e-9:
            best_total_cost = total_cost
            best_result = result.copy()
        
        # Perturbation
        num_perturb = 5
        for _ in range(num_perturb):
            i = np.random.randint(n_customers)
            old_f = result[i]
            feasible = [f for f in range(n_facilities) if f != old_f and load[f] + demands[i] <= capacities[f]]
            if feasible:
                new_f = np.random.choice(feasible)
                result[i] = new_f
                load[old_f] -= demands[i]
                load[new_f] += demands[i]
                count[old_f] -= 1
                count[new_f] += 1
        
        # Recalculate total cost after perturbation
        open_mask = (count > 0)
        total_cost = np.sum(open_cost[open_mask]) + np.sum(assign_cost[np.arange(n_customers), result])
        
        if time.time() - start_time > time_limit - 5.0:
            break

    return best_result
# EVOLVE-BLOCK-END
