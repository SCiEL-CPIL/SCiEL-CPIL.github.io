import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    best_result = None
    best_score = float('inf')
    
    # Function to evaluate a solution
    def evaluate(result):
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            load[f] += demands[i]
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    # Greedy Initialization with optional randomization
    def greedy_init(randomize=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Feasible facilities
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Should not happen if total capacity >= total demand, but handle gracefully
                # Fall back to cheapest
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Calculate marginal costs
            # If facility is open, cost is assign_cost[i, f]
            # If closed, cost is assign_cost[i, f] + open_cost[f]
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            # Mask infeasible
            marginal_costs[~feasible] = np.inf
            
            if randomize:
                # Pick from top 3
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                f = top_k[np.random.randint(len(top_k))]
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    # Local Search with 1-moves and 2-swaps
    def local_search(result, load, is_open):
        improved = True
        # We track the current cost to avoid recalculating from scratch if possible, 
        # but recalculating open costs is O(F) which is small.
        # Let's just use a counter for passes.
        
        while improved:
            improved = False
            
            # 1. Try 1-moves (relocation)
            for i in range(n_customers):
                f_old = result[i]
                d = demands[i]
                
                best_delta = 1e-9
                best_f_new = -1
                
                for f_new in range(n_facilities):
                    if f_new == f_old:
                        continue
                    
                    # Check capacity
                    if load[f_new] + d > capacities[f_new]:
                        continue
                    
                    # Calculate delta
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    delta_open = 0.0
                    # Check if old facility becomes empty
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]
                        
                    # Check if new facility was empty
                    if load[f_new] == 0:
                        if not is_open[f_new]:
                            delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                        
                if best_f_new != -1:
                    # Apply move
                    load[f_old] -= d
                    if load[f_old] == 0:
                        is_open[f_old] = False
                        
                    load[best_f_new] += d
                    if load[best_f_new] == d: # Was 0
                        is_open[best_f_new] = True
                        
                    result[i] = best_f_new
                    improved = True
                    
                    if time.time() > deadline:
                        return result, load, is_open

            # 2. Try 2-swaps (interchange)
            # To save time, we can limit this to pairs that are "close" in cost or just all pairs since N=100 is small.
            for i in range(n_customers):
                if time.time() > deadline:
                    return result, load, is_open
                for j in range(i + 1, n_customers):
                    f_i = result[i]
                    f_j = result[j]
                    if f_i == f_j:
                        continue
                    
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    # Check capacity feasibility for swap
                    # i moves to f_j, j moves to f_i
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    
                    # Calculate assignment cost delta
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    # Calculate open cost delta
                    # Determine new open status for f_i and f_j
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    
                    old_open_i = is_open[f_i]
                    old_open_j = is_open[f_j]
                    
                    new_open_i = (new_load_i > 0)
                    new_open_j = (new_load_j > 0)
                    
                    delta_open = 0.0
                    if old_open_i and not new_open_i:
                        delta_open -= open_cost[f_i]
                    elif not old_open_i and new_open_i:
                        delta_open += open_cost[f_i]
                        
                    if old_open_j and not new_open_j:
                        delta_open -= open_cost[f_j]
                    elif not old_open_j and new_open_j:
                        delta_open += open_cost[f_j]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < -1e-9: # Strict improvement
                        # Apply swap
                        result[i] = f_j
                        result[j] = f_i
                        
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        
                        if new_load_i == 0:
                            is_open[f_i] = False
                        else:
                            is_open[f_i] = True
                            
                        if new_load_j == 0:
                            is_open[f_j] = False
                        else:
                            is_open[f_j] = True
                            
                        improved = True
                        
                        if time.time() > deadline:
                            return result, load, is_open
                            
        return result, load, is_open

    # Main loop with restarts
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(randomize=False)
    res, load, is_open = local_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()
        
    # 2. Randomized restarts
    # The 2-swap local search is more expensive than 1-move.
    # N=100, pairs=5000. Each check is O(1). 5000 ops is negligible.
    # So we can afford more restarts.
    for k in range(100):
        if time.time() > deadline - 5: # Leave 5s buffer
            break
            
        res, load, is_open = greedy_init(randomize=True, seed=k)
        res, load, is_open = local_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
            
    return best_result
