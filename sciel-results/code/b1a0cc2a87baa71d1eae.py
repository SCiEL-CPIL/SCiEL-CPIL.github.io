# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def _greedy_init(demands, capacities, assign_cost, allowed, order=None):
    """
    Greedy assignment: process customers in given order, assign to cheapest feasible facility.
    Returns (assignment array, total_assign_cost) or (None, inf) if infeasible.
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    if n_fac == 0:
        return None, float('inf')
    
    if order is None:
        order = np.argsort(-demands)
    
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1)
    
    costs_for_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_for_allowed, axis=1)
    
    for i in order:
        placed = False
        for f_local in sorted_facs[i]:
            f_global = allowed[f_local]
            if load[f_local] + demands[i] <= capacities[f_global]:
                load[f_local] += demands[i]
                result[i] = f_global
                placed = True
                break
        if not placed:
            return None, float('inf')
    
    total_cost = float(np.sum([assign_cost[i, result[i]] for i in range(n_cust)]))
    return result, total_cost


def _local_search(result, demands, capacities, assign_cost, allowed, deadline=None, max_iters=10000):
    """
    Local search with single moves and swaps. Uses vectorized cost lookups.
    Modifies result in-place.
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    
    # Build load array and customer-to-local-index map
    load = np.zeros(n_fac)
    cust_local = np.empty(n_cust, dtype=int)
    for i in range(n_cust):
        f = result[i]
        idx = np.where(allowed == f)[0]
        if len(idx) == 0:
            continue
        cust_local[i] = idx[0]
        load[idx[0]] += demands[i]
    
    costs_allowed = assign_cost[:, allowed]
    caps = np.array([capacities[f] for f in allowed])
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        if deadline is not None and time.time() > deadline:
            break
        
        # Single moves
        for i in range(n_cust):
            ci = cust_local[i]
            for f_local in range(n_fac):
                if f_local == ci:
                    continue
                if load[f_local] + demands[i] <= caps[f_local]:
                    delta = costs_allowed[i, f_local] - costs_allowed[i, ci]
                    if delta < -1e-9:
                        load[ci] -= demands[i]
                        load[f_local] += demands[i]
                        cust_local[i] = f_local
                        result[i] = allowed[f_local]
                        improved = True
                        break
        
        # Swap moves
        for i in range(n_cust):
            ci = cust_local[i]
            for j in range(i + 1, n_cust):
                cj = cust_local[j]
                if ci == cj:
                    continue
                
                new_load_ci = load[ci] - demands[i] + demands[j]
                new_load_cj = load[cj] - demands[j] + demands[i]
                
                if new_load_ci <= caps[ci] and new_load_cj <= caps[cj]:
                    delta = (costs_allowed[i, cj] + costs_allowed[j, ci]) - \
                            (costs_allowed[i, ci] + costs_allowed[j, cj])
                    if delta < -1e-9:
                        load[ci] = new_load_ci
                        load[cj] = new_load_cj
                        cust_local[i] = cj
                        cust_local[j] = ci
                        result[i] = allowed[cj]
                        result[j] = allowed[ci]
                        improved = True
                        break
    
    return result


def _facility_close_move(result, demands, capacities, assign_cost, allowed, deadline=None):
    """
    Try closing a facility by moving all its customers to other facilities.
    Returns (new_result, cost_reduction) if successful, else (None, 0).
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    
    # Find which facilities are used
    used = set(result.tolist())
    if len(used) < 2:
        return None, 0
    
    best_reduction = 0
    best_new_result = None
    
    for f_to_close in used:
        if deadline is not None and time.time() > deadline:
            break
        
        # Find customers assigned to f_to_close
        custs_to_move = [i for i in range(n_cust) if result[i] == f_to_close]
        if not custs_to_move:
            continue
        
        # Try to move each customer to another facility
        new_result = result.copy()
        # Temporarily remove f_to_close from allowed for this check
        other_facilities = [f for f in used if f != f_to_close]
        if not other_facilities:
            continue
        
        # Build load for other facilities (excluding f_to_close)
        load_other = {f: 0 for f in other_facilities}
        for i in range(n_cust):
            if result[i] != f_to_close:
                load_other[result[i]] += demands[i]
        
        feasible = True
        cost_delta = 0
        for i in custs_to_move:
            best_target = None
            best_cost = float('inf')
            for f in other_facilities:
                if load_other[f] + demands[i] <= capacities[f]:
                    c = assign_cost[i, f]
                    if c < best_cost:
                        best_cost = c
                        best_target = f
            if best_target is None:
                feasible = False
                break
            load_other[best_target] += demands[i]
            cost_delta += best_cost - assign_cost[i, f_to_close]
            new_result[i] = best_target
        
        if feasible:
            # Cost reduction = open_cost of closed facility - additional assignment cost
            reduction = capacities[f_to_close] * 0  # placeholder
            # Actually, we need to know open_cost. Let's compute net change.
            # The caller will compute the full objective.
            if cost_delta < best_reduction or best_new_result is None:
                best_reduction = cost_delta
                best_new_result = new_result
    
    return best_new_result, best_reduction


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Multi-start greedy + local search with facility closing.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Multi-start: try different greedy orderings
    best_result = None
    best_total = float('inf')
    
    # Start 1: Sort by demand descending
    result1, cost1 = _greedy_init(demands, capacities, assign_cost, range(n_fac))
    if result1 is not None:
        _local_search(result1, demands, capacities, assign_cost, range(n_fac), deadline=deadline)
        used1 = set(result1.tolist())
        total1 = sum(open_cost[f] for f in used1) + sum(assign_cost[i, result1[i]] for i in range(n_cust))
        if total1 < best_total:
            best_total = total1
            best_result = result1.copy()
    
    # Start 2: Sort by assignment cost to closest facility (ascending)
    closest_cost = assign_cost.min(axis=1)
    order2 = np.argsort(closest_cost)
    result2, cost2 = _greedy_init(demands, capacities, assign_cost, range(n_fac), order=order2)
    if result2 is not None:
        _local_search(result2, demands, capacities, assign_cost, range(n_fac), deadline=deadline)
        used2 = set(result2.tolist())
        total2 = sum(open_cost[f] for f in used2) + sum(assign_cost[i, result2[i]] for i in range(n_cust))
        if total2 < best_total:
            best_total = total2
            best_result = result2.copy()
    
    # Start 3: Random order
    if time.time() < deadline - 20:
        result3, cost3 = _greedy_init(demands, capacities, assign_cost, range(n_fac), order=np.random.permutation(n_cust))
        if result3 is not None:
            _local_search(result3, demands, capacities, assign_cost, range(n_fac), deadline=deadline)
            used3 = set(result3.tolist())
            total3 = sum(open_cost[f] for f in used3) + sum(assign_cost[i, result3[i]] for i in range(n_cust))
            if total3 < best_total:
                best_total = total3
                best_result = result3.copy()
    
    # Facility closing: try to close facilities in the best result
    if best_result is not None:
        used = set(best_result.tolist())
        if len(used) > 1:
            # Try closing each facility one at a time
            for f_to_close in list(used):
                if time.time() > deadline - 5:
                    break
                
                custs_to_move = [i for i in range(n_cust) if best_result[i] == f_to_close]
                if not custs_to_move:
                    continue
                
                other_facilities = [f for f in used if f != f_to_close]
                load_other = {f: 0 for f in other_facilities}
                for i in range(n_cust):
                    if best_result[i] != f_to_close:
                        load_other[best_result[i]] += demands[i]
                
                new_result = best_result.copy()
                feasible = True
                cost_delta = 0
                for i in custs_to_move:
                    best_target = None
                    best_cost = float('inf')
                    for f in other_facilities:
                        if load_other[f] + demands[i] <= capacities[f]:
                            c = assign_cost[i, f]
                            if c < best_cost:
                                best_cost = c
                                best_target = f
                    if best_target is None:
                        feasible = False
                        break
                    load_other[best_target] += demands[i]
                    cost_delta += best_cost - assign_cost[i, f_to_close]
                    new_result[i] = best_target
                
                if feasible:
                    new_used = set(new_result.tolist())
                    new_total = sum(open_cost[f] for f in new_used) + sum(assign_cost[i, new_result[i]] for i in range(n_cust))
                    if new_total < best_total:
                        best_total = new_total
                        best_result = new_result
    
    # Fallback
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
    
    return best_result
# EVOLVE-BLOCK-END
