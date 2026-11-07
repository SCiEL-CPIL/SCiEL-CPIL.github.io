"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline to be safe within 150s
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def greedy_assignment():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    cost = assign_cost[i, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f == -1:
                best_f = 0
            result[i] = best_f
            load[best_f] += demands[i]
        return result, load

    def total_cost(result):
        cost = 0.0
        used = set(result)
        for i in range(n_customers):
            f = result[i]
            cost += assign_cost[i, f]
        for f in used:
            cost += open_cost[f]
        return cost

    def simulated_annealing(result, load, current_cost):
        # Initialize SA parameters
        temp = 50.0
        cooling_rate = 0.9995
        min_temp = 0.1
        max_iterations = 500000
        
        used = set(result)
        best_result = result.copy()
        best_cost = current_cost
        best_load = load.copy()
        
        iters = 0
        last_improvement = 0
        
        while temp > min_temp and iters < max_iterations:
            if time.time() > deadline:
                break
            
            # Pick a random customer
            i = np.random.randint(n_customers)
            old_f = result[i]
            
            # Pick a random feasible new facility
            feasible_facilities = [f for f in range(n_facilities) if f != old_f and load[f] + demands[i] <= capacities[f]]
            if not feasible_facilities:
                iters += 1
                continue
                
            new_f = random.choice(feasible_facilities)
            
            # Calculate delta cost
            delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
            
            delta_open = 0.0
            # Check if old facility becomes empty
            if load[old_f] == demands[i]:
                delta_open -= open_cost[old_f]
            # Check if new facility was empty
            if new_f not in used:
                delta_open += open_cost[new_f]
            
            delta_cost = delta_assign + delta_open
            new_cost = current_cost + delta_cost
            
            # Acceptance criterion
            accept = False
            if delta_cost < 0:
                accept = True
            else:
                # Metropolis criterion
                if temp > 0:
                    prob = np.exp(-delta_cost / temp)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                load[old_f] -= demands[i]
                load[new_f] += demands[i]
                result[i] = new_f
                
                if load[old_f] == 0:
                    used.discard(old_f)
                used.add(new_f)
                
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                    last_improvement = iters
            
            iters += 1
            temp *= cooling_rate
            
            # If no improvement for a long time, restart with higher temp
            if iters - last_improvement > 50000:
                temp = 50.0
                result = best_result.copy()
                load = best_load.copy()
                used = set(result)
                current_cost = best_cost
                last_improvement = iters
                
        return best_result, best_cost, best_load

    # Start with greedy
    result, load = greedy_assignment()
    current_cost = total_cost(result)
    
    # Run SA
    result, current_cost, load = simulated_annealing(result, load, current_cost)
    
    # Final local search polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_customers):
            old_f = result[i]
            for new_f in range(n_facilities):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                delta_open = 0.0
                if load[old_f] == demands[i]:
                    delta_open -= open_cost[old_f]
                if new_f not in set(result):
                    delta_open += open_cost[new_f]
                
                delta_cost = delta_assign + delta_open
                if delta_cost < -1e-9:
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    current_cost += delta_cost
                    improved = True
                    break
            if improved:
                break
    
    # Verify the result is valid
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            result, _ = greedy_assignment()
            break
        load_check[f] += demands[i]
    
    if np.any(load_check > capacities):
        result, _ = greedy_assignment()
    
    result = result.astype(int)
    return result
# EVOLVE-BLOCK-END
