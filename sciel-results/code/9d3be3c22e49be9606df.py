"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below 150s
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    def time_exceeded():
        return time.time() - start_time > timeout
    
    def greedy_init():
        """Largest demand first greedy initialization."""
        order = np.argsort(-demands)
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        
        for i in order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                f = np.argmax(remaining_cap)
            else:
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                f = np.argmin(costs)
            
            result[i] = f
            load[f] += d
        
        return result, load
    
    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum
    
    def local_search(result, load, max_rounds=50):
        """Best-improvement local search with 1-moves and 2-swaps."""
        current_cost = calc_total_cost(result, load)
        
        for _ in range(max_rounds):
            if time_exceeded():
                break
            
            improved = False
            best_delta = 0.0
            best_move_type = None
            best_move_data = None
            
            # --- Phase A: 1-moves (Relocation) ---
            for i in range(n_customers):
                if time_exceeded():
                    break
                
                f_old = result[i]
                d = demands[i]
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    if load[f_new] + d > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    
                    if load[f_old] == d:
                        delta_open -= open_cost[f_old]
                    
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move_type = 1
                        best_move_data = (i, f_old, f_new)
            
            # --- Phase B: 2-moves (Swap) ---
            for i in range(n_customers):
                if time_exceeded():
                    break
                
                f_i = result[i]
                d_i = demands[i]
                
                for j in range(i + 1, n_customers):
                    if time_exceeded():
                        break
                    
                    f_j = result[j]
                    d_j = demands[j]
                    
                    if f_i == f_j:
                        continue
                    
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                                   (assign_cost[i, f_i] + assign_cost[j, f_j])
                    
                    old_open = (open_cost[f_i] if load[f_i] > 0 else 0.0) + \
                               (open_cost[f_j] if load[f_j] > 0 else 0.0)
                    
                    new_open = (open_cost[f_i] if new_load_i > 0 else 0.0) + \
                               (open_cost[f_j] if new_load_j > 0 else 0.0)
                    
                    delta_open = new_open - old_open
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move_type = 2
                        best_move_data = (i, j, f_i, f_j)
            
            # Apply best move if found
            if best_move_type is not None:
                if best_move_type == 1:
                    i, f_old, f_new = best_move_data
                    load[f_old] -= demands[i]
                    load[f_new] += demands[i]
                    result[i] = f_new
                else:
                    i, j, f_i, f_j = best_move_data
                    d_i = demands[i]
                    d_j = demands[j]
                    load[f_i] -= d_i
                    load[f_j] -= d_j
                    load[f_i] += d_j
                    load[f_j] += d_i
                    result[i] = f_j
                    result[j] = f_i
                
                current_cost += best_delta
                improved = True
                
                if time_exceeded():
                    break
            else:
                break
        
        return result, load, current_cost
    
    # Initial greedy solution
    result, load = greedy_init()
    best_result, best_load, best_cost = local_search(result, load)
    
    # Random restarts with perturbations to escape local optima
    rng = np.random.RandomState(42)
    
    while not time_exceeded():
        # Create a perturbed version of the best solution
        result = best_result.copy()
        load = best_load.copy()
        
        # Perturb: randomly reassign 5-15 customers
        n_perturb = rng.randint(5, 16)
        indices = rng.choice(n_customers, size=n_perturb, replace=False)
        
        for i in indices:
            f_old = result[i]
            d = demands[i]
            load[f_old] -= d
            
            # Find a feasible facility
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if np.any(feasible_mask):
                feasible_indices = np.where(feasible_mask)[0]
                f_new = feasible_indices[rng.randint(0, len(feasible_indices))]
            else:
                f_new = np.argmax(remaining_cap)
            
            result[i] = f_new
            load[f_new] += d
        
        # Run local search on perturbed solution
        res, lod, cost = local_search(result, load)
        
        if cost < best_cost:
            best_result = res
            best_load = lod
            best_cost = cost
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
