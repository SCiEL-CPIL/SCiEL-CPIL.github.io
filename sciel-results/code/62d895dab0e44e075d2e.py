# EVOLVE-BLOCK-START
import numpy as np
import time

def _total_cost(result, demands, capacities, open_cost, assign_cost):
    used = np.unique(result)
    open_sum = float(open_cost[used].sum())
    assign_sum = float(assign_cost[np.arange(len(demands)), result].sum())
    return open_sum + assign_sum

def _is_valid(result, demands, capacities):
    load = np.zeros(len(capacities), dtype=float)
    for i in range(len(demands)):
        load[result[i]] += demands[i]
    return np.all(load <= capacities + 1e-9)

def _greedy_init(demands, capacities, open_cost, assign_cost, rng=None):
    n = len(demands); m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m, dtype=float)
    # Sort by demand descending
    order = np.argsort(-demands)
    
    if rng is None:
        rng = np.random.default_rng()

    for i in order:
        # Find feasible facilities
        feasible = (load + demands[i]) <= capacities
        if not np.any(feasible):
            # Fallback: pick facility with most remaining capacity (shouldn't happen if total cap > total demand)
            best_f = int(np.argmax(capacities - load))
        else:
            feas_idx = np.where(feasible)[0]
            # Calculate cost: assignment cost + marginal open cost
            costs = assign_cost[i, feas_idx]
            # Add open cost if facility is currently empty
            for f in feas_idx:
                if load[f] == 0:
                    costs[np.where(feas_idx == f)[0][0]] += open_cost[f]
            
            # RCL: Select from top alpha% of lowest costs or random within threshold
            min_cost = np.min(costs)
            # Dynamic alpha or fixed? Let's use a threshold based on min_cost
            # For diversification, we can pick randomly from the best few
            # Simple diversification: pick randomly from facilities with cost <= min_cost + (max_cost - min_cost)*0.1
            max_cost_in_feas = np.max(costs)
            threshold = min_cost + 0.2 * (max_cost_in_feas - min_cost)
            candidates = feas_idx[costs <= threshold]
            if len(candidates) == 0:
                candidates = feas_idx
            best_f = int(rng.choice(candidates))
        
        result[i] = best_f
        load[best_f] += demands[i]
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
    n = len(demands); m = len(capacities)
    result = result.copy()
    load = np.zeros(m, dtype=float)
    for i in range(n):
        load[result[i]] += demands[i]
    current_cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    best_cost = current_cost
    best_result = result.copy()
    start = time.time()
    while time.time() - start < time_limit:
        improved = False
        # Randomize order of customers to break symmetry/tie-breaking?
        # For determinism in LS steps, we iterate 0..n-1. 
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            cur_f = result[i]; d = demands[i]
            feasible = (load + d) <= capacities
            feasible[cur_f] = False
            if not np.any(feasible):
                continue
            feas_idx = np.where(feasible)[0]
            d_assign = assign_cost[i, feas_idx] - assign_cost[i, cur_f]
            d_open_t = np.where(load[feas_idx] == 0, open_cost[feas_idx], 0.0)
            d_open_s = -open_cost[cur_f] if (load[cur_f] - d) == 0 else 0.0
            total_delta = d_assign + d_open_t + d_open_s
            min_idx = np.argmin(total_delta)
            if total_delta[min_idx] < -1e-9:
                best_f = feas_idx[min_idx]
                load[cur_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += total_delta[min_idx]
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
    n = len(demands); m = len(capacities)
    global_start = time.time()
    deadline = global_start + 140.0 # Safety margin
    best_result = None
    best_cost = np.inf
    
    # Deterministic Greedy first
    g_res = _greedy_init(demands, capacities, open_cost, assign_cost)
    if _is_valid(g_res, demands, capacities):
        g_cost = _total_cost(g_res, demands, capacities, open_cost, assign_cost)
        best_cost = g_cost
        best_result = g_res.copy()
    
    # GRASP Loop
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        # Generate diverse initial solution
        init_res = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        
        if not _is_valid(init_res, demands, capacities):
            # Repair if needed, though greedy should be feasible if total capacity is sufficient
            # If not, skip or force repair
            continue
            
        # Local Search
        # Allocate time for LS: if we have 5s left, give 4s to LS
        time_left = deadline - time.time() - 1.0
        if time_left < 1.0:
            break
            
        # Limit LS time per iteration to ensure multiple iterations
        # If we have lots of time, do longer LS. If little, shorter.
        ls_time = min(10.0, time_left * 0.8)
        
        ls_res, ls_cost = _local_search(init_res, demands, capacities, open_cost, assign_cost, ls_time)
        
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_result = ls_res.copy()
            
    # Final fallback check
    if best_result is None or not _is_valid(best_result, demands, capacities):
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
        
    return best_result.astype(int)
# EVOLVE-BLOCK-END
