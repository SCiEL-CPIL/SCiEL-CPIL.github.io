# EVOLVE-BLOCK-START
import numpy as np
import time

def _total_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute objective: open cost of used facilities + assignment costs."""
    used = np.unique(result)
    open_sum = float(open_cost[used].sum())
    assign_sum = float(assign_cost[np.arange(len(demands)), result].sum())
    return open_sum + assign_sum

def _is_valid(result, demands, capacities):
    """Check capacity feasibility."""
    load = np.zeros(len(capacities), dtype=float)
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    return np.all(load <= capacities + 1e-9)

def _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=0):
    """Generate an initial feasible assignment.
    mode 0: largest demand first, cheapest facility.
    mode 1: random order, cheapest facility.
    mode 2: largest demand first, cheapest facility with open-cost penalty.
    """
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    
    if mode == 0:
        order = np.argsort(-demands)
    elif mode == 1:
        order = rng.permutation(n)
    else:
        order = np.argsort(-demands)
    
    for i in order:
        # Build candidate facility list
        feasible = []
        for f in range(m):
            if load[f] + demands[i] <= capacities[f]:
                if mode == 2:
                    # Penalize opening a new facility
                    penalty = open_cost[f] if load[f] == 0 else 0.0
                    feasible.append((assign_cost[i, f] + penalty * 0.1, f))
                else:
                    feasible.append((assign_cost[i, f], f))
        if not feasible:
            # Should not happen if capacities are sufficient; fallback
            feasible = [(assign_cost[i, f], f) for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if not feasible:
                # Force assign to least loaded
                f = int(np.argmin(load))
                result[i] = f
                load[f] += demands[i]
                continue
        feasible.sort()
        f = feasible[0][1]
        result[i] = f
        load[f] += demands[i]
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, rng, time_limit):
    """Improve assignment via vectorized best-improvement moves."""
    n = len(demands)
    m = len(capacities)
    result = result.copy()
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Vectorized move evaluation
        # For each customer i, we want to find the best facility f
        # such that load[f] + demands[i] <= capacities[f]
        # and delta_cost < 0.
        
        # Current assignment costs
        cur_facs = result
        cur_assign_costs = assign_cost[np.arange(n), cur_facs]
        
        # We iterate through customers to find the best global move
        # To keep it fast, we do a single pass of best-improvement moves
        # We can vectorize the cost calculation for all potential moves for all customers
        # but capacity constraints make it tricky. 
        # Simpler: Loop over customers, but use numpy for facility comparison.
        
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            
            cur_f = result[i]
            d = demands[i]
            
            # Feasible facilities: load[f] + d <= capacities[f]
            feasible_mask = (load + d) <= capacities
            feasible_mask[cur_f] = False # Exclude current
            
            if not np.any(feasible_mask):
                continue
            
            # Calculate delta for all feasible facilities
            # delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
            delta_assign = assign_cost[i, feasible_mask] - assign_cost[i, cur_f]
            
            # delta_open:
            # If we move i away from cur_f, cur_f might close.
            # If we move i to f, f might open.
            
            # Cost of opening cur_f if it becomes empty:
            # load[cur_f] - d > 0 ? 0 : open_cost[cur_f]
            new_load_cur = load[cur_f] - d
            old_open_cur = 0.0 if new_load_cur > 0 else open_cost[cur_f]
            # Wait, current cost includes open_cost[cur_f] if load[cur_f] > 0.
            # If load[cur_f] == d, it becomes 0, so we save open_cost[cur_f].
            # If load[cur_f] > d, it stays open, so no change.
            # So delta_open_cur = -open_cost[cur_f] if new_load_cur == 0 else 0.0
            
            # For target f:
            # If load[f] == 0, it opens. Cost +open_cost[f].
            # If load[f] > 0, it stays open. Cost 0.
            
            # We need to compute delta_open for each feasible f.
            # This is hard to vectorize directly with the "if load[f]==0" condition 
            # combined with the specific cur_f logic without masking.
            
            # Let's just compute the total new cost for each feasible f and compare to current total cost contribution?
            # No, total cost is global.
            
            # Let's stick to the previous logic but ensure it's fast enough.
            # The previous logic was O(M) per customer. 100 * 20 = 2000 ops. 
            # The issue was likely the SWAP loop O(N^2).
            # By removing the swap loop, we reduce complexity significantly.
            
            best_f = -1
            best_delta = 0
            
            feasible_indices = np.where(feasible_mask)[0]
            
            # Vectorized calculation of delta for feasible indices
            target_facs = feasible_indices
            target_loads = load[target_facs]
            
            # delta_assign for these targets
            d_assign = assign_cost[i, target_facs] - assign_cost[i, cur_f]
            
            # delta_open for these targets
            # If target_load == 0, we pay open_cost[target_f]. Else 0.
            d_open_target = np.where(target_loads == 0, open_cost[target_facs], 0.0)
            
            # delta_open for source cur_f
            # If (load[cur_f] - d) == 0, we save open_cost[cur_f]. Else 0.
            d_open_source = -open_cost[cur_f] if (load[cur_f] - d) == 0 else 0.0
            
            total_delta = d_assign + d_open_target + d_open_source
            
            # Find min delta
            min_idx = np.argmin(total_delta)
            min_delta = total_delta[min_idx]
            
            if min_delta < -1e-9:
                best_f = target_facs[min_idx]
                
                # Apply move
                load[cur_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += min_delta
                improved = True
                
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_result = result.copy()
        
        if not improved:
            break
    
    return best_result, best_cost

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    # Deadline guard: keep safety margin of 20s before 150s limit
    global_start = time.time()
    deadline = global_start + 100.0  # 50s safety margin
    
    rng = np.random.RandomState(42)
    
    best_result = None
    best_cost = np.inf
    
    # Multi-start with different seeding modes
    for restart in range(20):
        if time.time() >= deadline:
            break
        
        # Choose seeding mode
        mode = restart % 3
        
        # Generate seed
        seed_result = _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=mode)
        
        # Validate
        if not _is_valid(seed_result, demands, capacities):
            continue
        
        seed_cost = _total_cost(seed_result, demands, capacities, open_cost, assign_cost)
        
        if seed_cost < best_cost:
            best_cost = seed_cost
            best_result = seed_result.copy()
        
        # Local search with time budget
        time_remaining = deadline - time.time()
        ls_time = min(2.0, time_remaining * 0.4)  # Up to 2s per local search
        if ls_time <= 0:
            break
        
        ls_result, ls_cost = _local_search(
            seed_result, demands, capacities, open_cost, assign_cost, rng, ls_time
        )
        
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_result = ls_result.copy()
    
    # If no valid result found, fall back to simple greedy
    if best_result is None:
        best_result = _greedy_seed(demands, capacities, open_cost, assign_cost, rng, mode=0)
    
    # Final validation
    if not _is_valid(best_result, demands, capacities):
        # Fallback: assign each customer to a facility with most remaining capacity
        load = np.zeros(m, dtype=float)
        best_result = np.zeros(n, dtype=int)
        for i in np.argsort(-demands):
            feasible = [f for f in range(m) if load[f] + demands[i] <= capacities[f]]
            if feasible:
                # Pick cheapest
                f = min(feasible, key=lambda x: assign_cost[i, x])
            else:
                f = int(np.argmin(load))
            best_result[i] = f
            load[f] += demands[i]
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
