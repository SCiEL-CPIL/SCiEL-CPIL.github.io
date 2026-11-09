import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Solution: Greedy (Largest Demand First, Cheapest Feasible)
    # This provides a good starting point for Simulated Annealing
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    order = np.argsort(-demands)
    for i in order:
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
        else:
            f_best = np.argmax(capacities - load)
            result[i] = f_best
            load[f_best] += demands[i]

    def get_cost(res, ld):
        total = 0.0
        for f in range(n_facilities):
            if ld[f] > 0:
                total += open_cost[f]
        total += np.sum(assign_cost[np.arange(n_customers), res])
        return total

    current_cost = get_cost(result, load)
    best_result = result.copy()
    best_load = load.copy()
    best_cost = current_cost

    start_time = time.time()
    time_limit = 15.0

    # Simulated Annealing Parameters
    # T0: Initial temperature, should be high enough to accept many bad moves
    # alpha: Cooling factor
    # T_end: Final temperature
    T0 = 100.0
    alpha = 0.9995
    T = T0
    
    # Precompute some structures for speed if needed, but n is small
    # We will use random moves and swaps
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        # 50% chance move, 50% chance swap
        if np.random.rand() < 0.5:
            # Move
            i = np.random.randint(n_customers)
            f_new = np.random.randint(n_facilities)
            f_old = result[i]
            if f_new == f_old:
                continue
            
            # Check capacity
            if load[f_new] + demands[i] > capacities[f_new]:
                continue
                
            # Calculate delta
            delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
            delta_open = 0.0
            
            if load[f_old] == demands[i]:
                delta_open -= open_cost[f_old]
            if load[f_new] == 0:
                delta_open += open_cost[f_new]
                
            delta = delta_assign + delta_open
            
            # Acceptance criterion
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                # Apply move
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
        else:
            # Swap
            i = np.random.randint(n_customers)
            j = np.random.randint(n_customers)
            if i == j:
                continue
                
            f_i = result[i]
            f_j = result[j]
            if f_i == f_j:
                continue
                
            # Check capacities
            if load[f_i] - demands[i] + demands[j] > capacities[f_i]:
                continue
            if load[f_j] - demands[j] + demands[i] > capacities[f_j]:
                continue
                
            # Calculate delta
            # Swaps do not change open status if both facilities have other customers
            # But if one facility had only this customer, it might close?
            # No, if we swap i from f_i to f_j, and j from f_j to f_i:
            # f_i loses i, gains j. f_j loses j, gains i.
            # If load[f_i] was demands[i], it becomes demands[j]. If demands[j]>0, it stays open.
            # If load[f_j] was demands[j], it becomes demands[i]. If demands[i]>0, it stays open.
            # So open status is invariant for swaps where demands > 0.
            delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
            
            # Acceptance criterion
            if delta_assign < 0 or np.random.rand() < np.exp(-delta_assign / T):
                # Apply swap
                load[f_i] -= demands[i]
                load[f_j] -= demands[j]
                load[f_i] += demands[j]
                load[f_j] += demands[i]
                result[i] = f_j
                result[j] = f_i
                current_cost += delta_assign
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                    
        # Cool down
        T *= alpha
        if T < 1e-6:
            T = 1e-6

    return best_result
