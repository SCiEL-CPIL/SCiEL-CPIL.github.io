import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    # Safety margin: 150s limit, use 145s to be safe
    deadline = start_time + 145.0
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Ensure inputs are numpy arrays for speed
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    best_result = None
    best_score = float('inf')
    
    def evaluate(result):
        """Calculate total cost: open_cost + assign_cost"""
        load = np.zeros(n_facilities)
        for i, f in enumerate(result):
            if f < 0 or f >= n_facilities:
                return float('inf')
            load[f] += demands[i]
        
        if np.any(load > capacities + 1e-9):
            return float('inf')
        
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in result:
            open_mask[f] = True
        
        open_sum = np.sum(open_cost[open_mask])
        assign_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_sum + assign_sum

    def greedy_init(seed=None):
        """Greedy construction heuristic"""
        if seed is not None:
            np.random.seed(seed)
        
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        is_open = np.zeros(n_facilities, dtype=bool)
        
        # Sort customers by demand descending to fill large facilities first
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            feasible = load + d <= capacities
            
            if not np.any(feasible):
                # Fallback: pick cheapest feasible (should not happen if total cap >= total demand)
                # If truly infeasible, pick argmin assign_cost
                f = np.argmin(assign_cost[i])
                result[i] = f
                load[f] += d
                is_open[f] = True
                continue
            
            # Marginal cost: assign_cost + open_cost if facility is closed
            marginal_costs = assign_cost[i].copy()
            closed_mask = ~is_open
            marginal_costs[closed_mask] += open_cost[closed_mask]
            
            # Mask infeasible
            marginal_costs[~feasible] = np.inf
            
            # Randomization: pick from top 3 feasible with lowest marginal cost
            if np.random.rand() < 0.3: # 30% chance to randomize slightly
                sorted_indices = np.argsort(marginal_costs)
                top_k = sorted_indices[:min(3, n_facilities)]
                # Filter out infeasible just in case
                valid_top = [idx for idx in top_k if feasible[idx]]
                if valid_top:
                    f = valid_top[np.random.randint(len(valid_top))]
                else:
                    f = np.argmin(marginal_costs)
            else:
                f = np.argmin(marginal_costs)
            
            result[i] = f
            load[f] += d
            is_open[f] = True
            
        return result, load, is_open

    def local_search(result, load, is_open):
        """
        Alternating 1-move and 2-swap local search.
        1-moves can change open status.
        2-swaps do NOT change open status (swapping between two open facilities).
        """
        improved = True
        # Limit passes to prevent infinite loops in pathological cases, though convergence is expected
        max_passes = 100 
        
        while improved and max_passes > 0:
            improved = False
            max_passes -= 1
            
            # Check time
            if time.time() > deadline:
                return result, load, is_open

            # --- Phase 1: 1-Move Optimization (Relocation) ---
            # This phase considers open costs.
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
                    
                    # Calculate delta in assignment cost
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    
                    # Calculate delta in open cost
                    delta_open = 0.0
                    
                    # If old facility becomes empty, we save its open cost
                    if load[f_old] - d == 0:
                        delta_open -= open_cost[f_old]
                        
                    # If new facility was empty (closed), we incur its open cost
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

            # --- Phase 2: 2-Swap Optimization (Interchange) ---
            # This phase only considers assignment costs, as open status is invariant.
            # We iterate until no swap improves the assignment cost.
            swap_improved = True
            while swap_improved:
                swap_improved = False
                if time.time() > deadline:
                    return result, load, is_open

                for i in range(n_customers):
                    f_i = result[i]
                    d_i = demands[i]
                    
                    for j in range(i + 1, n_customers):
                        f_j = result[j]
                        d_j = demands[j]
                        
                        if f_i == f_j:
                            continue
                        
                        # Check capacity feasibility for swap
                        # f_i loses d_i, gains d_j
                        # f_j loses d_j, gains d_i
                        if load[f_i] - d_i + d_j > capacities[f_i]:
                            continue
                        if load[f_j] - d_j + d_i > capacities[f_j]:
                            continue
                        
                        # Calculate assignment cost delta
                        delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                       (assign_cost[j, f_i] - assign_cost[j, f_j])
                        
                        # Open cost delta is 0 because:
                        # f_i had customer i, now has customer j. Still non-empty.
                        # f_j had customer j, now has customer i. Still non-empty.
                        # So open status of f_i and f_j does not change.
                        
                        if delta_assign < -1e-9: # Strict improvement
                            # Apply swap
                            result[i] = f_j
                            result[j] = f_i
                            
                            load[f_i] -= d_i
                            load[f_i] += d_j
                            load[f_j] -= d_j
                            load[f_j] += d_i
                            
                            swap_improved = True
                            improved = True # Mark overall improved to trigger another 1-move pass if needed
                            
                            if time.time() > deadline:
                                return result, load, is_open
                                
        return result, load, is_open

    # Main Loop: Multi-Start
    # 1. Deterministic greedy
    res, load, is_open = greedy_init(seed=0)
    res, load, is_open = local_search(res, load, is_open)
    score = evaluate(res)
    if score < best_score:
        best_score = score
        best_result = res.copy()
        
    # 2. Randomized restarts
    restart_id = 1
    while time.time() < deadline - 2.0: # Leave 2s buffer for final evaluation and return
        res, load, is_open = greedy_init(seed=restart_id)
        res, load, is_open = local_search(res, load, is_open)
        score = evaluate(res)
        if score < best_score:
            best_score = score
            best_result = res.copy()
            # If we found a very good solution, we can continue searching
            # Or we can break if we hit a theoretical lower bound? No, we don't know LB.
        restart_id += 1

    # Final safety check
    if best_result is None:
        # Fallback: assign everyone to facility 0 (might be infeasible, but better than None)
        best_result = np.zeros(n_customers, dtype=int)
        
    return best_result
