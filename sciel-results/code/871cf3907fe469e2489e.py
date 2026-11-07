"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe within 150s
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def greedy_assignment():
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
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
                # If no feasible facility, assign to facility with most remaining capacity
                remaining = capacities - load
                best_f = np.argmax(remaining)
                if load[best_f] + demands[i] > capacities[best_f]:
                    # Last resort: assign to facility 0 (shouldn't happen with proper capacities)
                    best_f = 0
            result[i] = best_f
            load[best_f] += demands[i]
        return result, load
    
    def total_cost(result):
        cost = 0.0
        used = set(result.tolist())
        for i in range(n_customers):
            f = result[i]
            cost += assign_cost[i, f]
        for f in used:
            cost += open_cost[f]
        return cost
    
    def local_search_1move(result, load, current_cost, time_limit):
        """Local search with 1-move improvements."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n_customers):
                old_f = result[i]
                best_delta = 0
                best_new_f = -1
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                    
                    delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                    delta_open = 0.0
                    # Check if old facility becomes empty
                    if load[old_f] == demands[i]:
                        delta_open -= open_cost[old_f]
                    # Check if new facility was empty
                    if load[new_f] == 0:
                        delta_open += open_cost[new_f]
                    
                    delta_cost = delta_assign + delta_open
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_new_f = new_f
                
                if best_new_f != -1:
                    # Apply the best move
                    load[old_f] -= demands[i]
                    load[best_new_f] += demands[i]
                    result[i] = best_new_f
                    current_cost += best_delta
                    improved = True
        
        return result, current_cost, load
    
    def local_search_2swap(result, load, current_cost, time_limit):
        """Local search with 2-swap improvements."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n_customers):
                if time.time() > time_limit:
                    break
                old_f_i = result[i]
                for j in range(i + 1, n_customers):
                    if time.time() > time_limit:
                        break
                    old_f_j = result[j]
                    if old_f_i == old_f_j:
                        continue
                    
                    # Try swapping: i -> old_f_j, j -> old_f_i
                    if load[old_f_j] - demands[j] + demands[i] <= capacities[old_f_j] and \
                       load[old_f_i] - demands[i] + demands[j] <= capacities[old_f_i]:
                        
                        delta_assign = (assign_cost[i, old_f_j] - assign_cost[i, old_f_i]) + \
                                       (assign_cost[j, old_f_i] - assign_cost[j, old_f_j])
                        
                        delta_open = 0.0
                        # After swap, facility old_f_i has load - demands[i] + demands[j]
                        # Facility old_f_j has load - demands[j] + demands[i]
                        # Check if either facility becomes empty
                        new_load_i = load[old_f_i] - demands[i] + demands[j]
                        new_load_j = load[old_f_j] - demands[j] + demands[i]
                        
                        if new_load_i == 0 and load[old_f_i] != 0:
                            delta_open -= open_cost[old_f_i]
                        if new_load_j == 0 and load[old_f_j] != 0:
                            delta_open -= open_cost[old_f_j]
                        if new_load_i > 0 and load[old_f_i] == 0:
                            delta_open += open_cost[old_f_i]
                        if new_load_j > 0 and load[old_f_j] == 0:
                            delta_open += open_cost[old_f_j]
                        
                        delta_cost = delta_assign + delta_open
                        if delta_cost < -1e-9:
                            # Apply swap
                            load[old_f_i] -= demands[i]
                            load[old_f_j] -= demands[j]
                            load[old_f_j] += demands[i]
                            load[old_f_i] += demands[j]
                            result[i] = old_f_j
                            result[j] = old_f_i
                            current_cost += delta_cost
                            improved = True
                            break
                if time.time() > time_limit:
                    break
        
        return result, current_cost, load
    
    def simulated_annealing(result, load, current_cost, time_limit):
        """Simulated annealing for global search."""
        temp = 30.0
        cooling_rate = 0.9998
        min_temp = 0.01
        max_iterations = 300000
        
        used = set(result.tolist())
        best_result = result.copy()
        best_cost = current_cost
        best_load = load.copy()
        
        iters = 0
        last_improvement = 0
        
        while temp > min_temp and iters < max_iterations:
            if time.time() > time_limit:
                break
            
            # Pick a random customer
            i = np.random.randint(n_customers)
            old_f = result[i]
            
            # Pick a random feasible new facility
            feasible = []
            for f in range(n_facilities):
                if f != old_f and load[f] + demands[i] <= capacities[f]:
                    feasible.append(f)
            
            if not feasible:
                iters += 1
                continue
                
            new_f = random.choice(feasible)
            
            # Calculate delta cost
            delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
            
            delta_open = 0.0
            if load[old_f] == demands[i]:
                delta_open -= open_cost[old_f]
            if load[new_f] == 0:
                delta_open += open_cost[new_f]
            
            delta_cost = delta_assign + delta_open
            new_cost = current_cost + delta_cost
            
            # Acceptance criterion
            accept = False
            if delta_cost < 0:
                accept = True
            else:
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
                if load[new_f] > 0:
                    used.add(new_f)
                
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = result.copy()
                    best_load = load.copy()
                    last_improvement = iters
            
            iters += 1
            temp *= cooling_rate
            
            # If no improvement for a long time, restart
            if iters - last_improvement > 30000:
                temp = 30.0
                result = best_result.copy()
                load = best_load.copy()
                used = set(result.tolist())
                current_cost = best_cost
                last_improvement = iters
                
        return best_result, best_cost, best_load
    
    # Start with greedy
    result, load = greedy_assignment()
    current_cost = total_cost(result)
    
    # Run local search 1-move
    time_limit_1 = start_time + 20
    result, current_cost, load = local_search_1move(result, load, current_cost, time_limit_1)
    
    # Run local search 2-swap
    time_limit_2 = start_time + 40
    result, current_cost, load = local_search_2swap(result, load, current_cost, time_limit_2)
    
    # Run simulated annealing
    time_limit_3 = deadline - 5
    result, current_cost, load = simulated_annealing(result, load, current_cost, time_limit_3)
    
    # Final local search polish
    time_limit_4 = deadline - 2
    result, current_cost, load = local_search_1move(result, load, current_cost, time_limit_4)
    result, current_cost, load = local_search_2swap(result, load, current_cost, time_limit_4)
    
    # Verify the result is valid
    load_check = np.zeros(n_facilities)
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            result, _ = greedy_assignment()
            load_check = np.zeros(n_facilities)
            for i in range(n_customers):
                load_check[result[i]] += demands[i]
            break
        load_check[f] += demands[i]
    
    if np.any(load_check > capacities + 1e-6):
        result, _ = greedy_assignment()
    
    result = result.astype(int)
    return result
# EVOLVE-BLOCK-END
