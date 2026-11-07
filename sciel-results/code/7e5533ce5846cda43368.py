# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, open_cost, assign_cost):
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open


def _repair(result, demands, capacities, assign_cost):
    """Fix capacity violations by moving customers from overloaded facilities to cheapest feasible ones."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    # Iteratively fix overloaded facilities
    for _ in range(200):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            # Find customers at f_over, move those with most savings to other facilities
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            # Sort by demand descending (move big ones first to reduce excess faster)
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
                # Find cheapest feasible facility (not f_over)
                best_f = -1
                best_cost = np.inf
                for f in range(m):
                    if f == f_over:
                        continue
                    if load[f] + d <= capacities[f] + 1e-6:
                        c = assign_cost[i, f]
                        if c < best_cost:
                            best_cost = c
                            best_f = f
                if best_f >= 0:
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
                else:
                    # No feasible facility, must move anyway (shouldn't happen if total cap >= total demand)
                    rem = capacities - load
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    
    return result


def _greedy_init(demands, capacities, open_cost, assign_cost, rng):
    """Greedy assignment with randomization: process in random order, pick cheapest feasible."""
    n = len(demands)
    m = len(capacities)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    
    order = rng.permutation(n)
    for i in order:
        d = demands[i]
        best_f = -1
        best_cost = np.inf
        for f in range(m):
            if load[f] + d <= capacities[f] + 1e-6:
                c = assign_cost[i, f]
                if c < best_cost:
                    best_cost = c
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += d
        else:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            best_f = int(np.argmax(rem))
            result[i] = best_f
            load[best_f] += d
    
    # Repair any remaining violations
    result = _repair(result, demands, capacities, assign_cost)
    return result


def _cheapest_init(demands, capacities, open_cost, assign_cost):
    """Assign each customer to their cheapest facility, then repair capacity violations."""
    n = len(demands)
    m = len(capacities)
    result = np.argmin(assign_cost, axis=1).astype(int)
    result = _repair(result, demands, capacities, assign_cost)
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, max_iters=500, deadline=None, rng=None):
    """Local search with single moves and 2-opt swaps."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(demands)
    m = len(capacities)
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(n)
        
        # Single moves
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f] + 1e-6:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == d:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        # 2-opt swaps: swap two customers between two different facilities
        if deadline is not None and time.time() > deadline:
            break
        
        swap_order = rng.permutation(n)
        for i in swap_order:
            if deadline is not None and time.time() > deadline:
                break
            fi = result[i]
            di = demands[i]
            
            for j in range(n):
                if j == i:
                    continue
                fj = result[j]
                if fj == fi:
                    continue
                dj = demands[j]
                
                # Check capacity: after swap, fi has load[fi] - di + dj, fj has load[fj] - dj + di
                if load[fi] - di + dj > capacities[fi] + 1e-6:
                    continue
                if load[fj] - dj + di > capacities[fj] + 1e-6:
                    continue
                
                delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes
                if load[fi] == di:  # fi becomes empty after removing i, but j comes in
                    # fi won't be empty since j is coming
                    pass
                if load[fj] == dj:  # fj becomes empty after removing j, but i comes in
                    pass
                
                # More careful: check if facility becomes empty or was empty
                # After swap: load[fi] becomes load[fi] - di + dj, load[fj] becomes load[fj] - dj + di
                new_load_fi = load[fi] - di + dj
                new_load_fj = load[fj] - dj + di
                
                # fi was empty? No, it has di >= 1
                # fi becomes empty? new_load_fi == 0 means di == dj and load[fi] == di
                if load[fi] == di and new_load_fi == 0:
                    delta -= open_cost[fi]
                if load[fi] > di and new_load_fi == 0:
                    delta -= open_cost[fi]
                
                # Simplified: just check if facility transitions to/from empty
                # fi: was load[fi], now new_load_fi
                if load[fi] > 0 and new_load_fi == 0:
                    delta -= open_cost[fi]
                elif load[fi] == 0 and new_load_fi > 0:
                    delta += open_cost[fi]
                
                # fj: was load[fj], now new_load_fj
                if load[fj] > 0 and new_load_fj == 0:
                    delta -= open_cost[fj]
                elif load[fj] == 0 and new_load_fj > 0:
                    delta += open_cost[fj]
                
                if delta < -1e-9:
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    result[i] = fj
                    result[j] = fi
                    current_cost += delta
                    improved = True
                    break  # restart inner loop after a swap
        
        if not improved:
            break
    
    return result, current_cost


def _local_search_fast(result, demands, capacities, open_cost, assign_cost, max_iters=200, deadline=None, rng=None):
    """Faster local search: only single moves, no swaps, for polish."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(demands)
    m = len(capacities)
    
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]
    
    current_cost = _compute_cost(result, demands, open_cost, assign_cost)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        order = rng.permutation(n)
        
        for i in order:
            current_f = result[i]
            d = demands[i]
            
            best_f = -1
            best_delta = 0.0
            
            for f in range(m):
                if f == current_f:
                    continue
                if load[f] + d > capacities[f] + 1e-6:
                    continue
                
                delta = assign_cost[i, f] - assign_cost[i, current_f]
                
                if load[current_f] == d:
                    delta -= open_cost[current_f]
                if load[f] == 0:
                    delta += open_cost[f]
                
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_f = f
            
            if best_f >= 0:
                load[current_f] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += best_delta
                improved = True
        
        if not improved:
            break
    
    return result, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n = len(demands)
    m = len(capacities)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: Multiple initializations
    # 1a: Cheapest assignment + repair
    result = _cheapest_init(demands, capacities, open_cost, assign_cost)
    result, cost = _local_search(result, demands, capacities, open_cost, assign_cost,
                                  max_iters=300, deadline=deadline, rng=rng)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()
    
    if time.time() > deadline - 20:
        return best_result
    
    # 1b: Greedy with different random orderings
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        result = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        result, cost = _local_search(result, demands, capacities, open_cost, assign_cost,
                                      max_iters=200, deadline=deadline, rng=rng)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    if time.time() > deadline - 10:
        return best_result
    
    # Phase 2: Polish the best result with more iterations
    result = best_result.copy()
    result, cost = _local_search(result, demands, capacities, open_cost, assign_cost,
                                  max_iters=300, deadline=deadline, rng=rng)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()
    
    if time.time() > deadline - 5:
        return best_result
    
    # Phase 3: Final polish with fast single-move search
    result = best_result.copy()
    result, cost = _local_search_fast(result, demands, capacities, open_cost, assign_cost,
                                       max_iters=100, deadline=deadline, rng=rng)
    if cost < best_cost:
        best_cost = cost
        best_result = result.copy()
    
    # Final validity check
    load = np.zeros(m)
    valid = True
    for i in range(n):
        f = best_result[i]
        if f < 0 or f >= m:
            valid = False
            break
        load[f] += demands[i]
    
    if not valid or np.any(load > capacities + 1e-6):
        # Fallback: greedy init
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
    
    return best_result
# EVOLVE-BLOCK-END
