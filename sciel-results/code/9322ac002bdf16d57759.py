# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def _min_cost_assignment(demands, capacities, assign_cost, allowed):
    """
    Solve capacitated assignment for customers to a fixed set of allowed facilities.
    Returns (assignment array, total_assign_cost) or (None, inf) if infeasible.
    Uses a priority-queue based greedy that assigns customers to the cheapest
    facility with remaining capacity, processing customers in order of largest
    demand first to reduce infeasibility risk.
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    if n_fac == 0:
        return None, float('inf')
    
    # Check total capacity
    if sum(capacities[f] for f in allowed) < sum(demands):
        return None, float('inf')
    
    # Greedy with priority queue: process largest demands first
    order = np.argsort(-demands)
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1)
    
    # Precompute sorted facility indices by cost for each customer
    costs_for_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_for_allowed, axis=1)
    
    feasible = True
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
            feasible = False
            break
    
    if not feasible:
        return None, float('inf')
    
    total_cost = sum(assign_cost[i, result[i]] for i in range(n_cust))
    return result, total_cost


def _multi_start_greedy(demands, capacities, assign_cost, allowed, n_trials=10):
    """
    Multi-start greedy: try different customer orderings to find feasible assignments.
    Returns (best_result, best_assign_cost) or (None, inf).
    """
    n_cust = len(demands)
    n_fac = len(allowed)
    
    costs_for_allowed = assign_cost[:, allowed]
    sorted_facs = np.argsort(costs_for_allowed, axis=1)
    
    best_result = None
    best_cost = float('inf')
    
    # First try: largest demand first (most likely to be feasible)
    orders = [np.argsort(-demands)]
    # Then random permutations
    for _ in range(n_trials - 1):
        orders.append(np.random.permutation(n_cust))
    
    for order in orders:
        load = np.zeros(n_fac)
        result = np.full(n_cust, -1)
        feasible = True
        
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
                feasible = False
                break
        
        if feasible:
            total_cost = sum(assign_cost[i, result[i]] for i in range(n_cust))
            if total_cost < best_cost:
                best_cost = total_cost
                best_result = result.copy()
    
    if best_result is None:
        return None, float('inf')
    
    return best_result, best_cost


def _local_search(result, demands, capacities, assign_cost, allowed_set, max_iters=5000, deadline=None):
    """
    Local search: try moving single customers and swapping pairs.
    Only consider facilities in allowed_set.
    Uses vectorized single-move pass for speed.
    """
    n_cust = len(demands)
    n_fac = len(allowed_set)
    allowed = np.array(sorted(allowed_set))
    
    # Build load array
    load = np.zeros(n_fac)
    fac_to_local = {f: idx for idx, f in enumerate(allowed)}
    for i in range(n_cust):
        f = result[i]
        if f in fac_to_local:
            load[fac_to_local[f]] += demands[i]
    
    # Precompute costs
    costs_allowed = assign_cost[:, allowed]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Vectorized single moves: for each customer, find best feasible move
        for i in range(n_cust):
            current_f = result[i]
            ci = fac_to_local.get(current_f)
            if ci is None:
                continue
            
            # Find all facilities with enough capacity
            feasible_mask = (load + demands[i]) <= capacities[allowed]
            feasible_mask[ci] = False  # skip current
            
            if not feasible_mask.any():
                continue
            
            feasible_indices = np.where(feasible_mask)[0]
            # Pick the one with lowest cost
            best_f_local = feasible_indices[np.argmin(costs_allowed[i, feasible_indices])]
            
            delta = costs_allowed[i, best_f_local] - costs_allowed[i, ci]
            if delta < -1e-9:
                load[ci] -= demands[i]
                load[best_f_local] += demands[i]
                result[i] = allowed[best_f_local]
                improved = True
        
        # Swap moves (sampled to keep runtime bounded)
        if not improved and iters < max_iters:
            # Try a subset of swaps for speed
            n_swap_trials = min(50, n_cust * (n_cust - 1) // 2)
            for _ in range(n_swap_trials):
                i = np.random.randint(0, n_cust)
                j = np.random.randint(0, n_cust)
                if i == j:
                    continue
                
                fi = result[i]
                fj = result[j]
                ci = fac_to_local.get(fi)
                cj = fac_to_local.get(fj)
                if ci is None or cj is None or ci == cj:
                    continue
                
                # Check if swap is feasible
                new_load_ci = load[ci] - demands[i] + demands[j]
                new_load_cj = load[cj] - demands[j] + demands[i]
                
                if new_load_ci <= capacities[allowed[ci]] and new_load_cj <= capacities[allowed[cj]]:
                    delta = (costs_allowed[i, cj] + costs_allowed[j, ci]) - \
                            (costs_allowed[i, ci] + costs_allowed[j, cj])
                    if delta < -1e-9:
                        load[ci] = new_load_ci
                        load[cj] = new_load_cj
                        result[i] = allowed[cj]
                        result[j] = allowed[ci]
                        improved = True
                        break


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses facility subset enumeration + multi-start greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Precompute: for each facility, compute a "value" metric
    avg_assign_cost = assign_cost.mean(axis=0)  # shape (n_fac,)
    
    # Generate candidate subsets of facilities
    candidate_subsets = []
    
    # All facilities
    candidate_subsets.append(set(range(n_fac)))
    
    # Sort by open cost
    by_open = np.argsort(open_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_open[:k].tolist()))
    
    # Sort by avg assign cost
    by_assign = np.argsort(avg_assign_cost)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_assign[:k].tolist()))
    
    # Sort by combined metric
    combined = open_cost + avg_assign_cost
    by_combined = np.argsort(combined)
    for k in [5, 8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_combined[:k].tolist()))
    
    # Sort by capacity-adjusted metric: prefer facilities with high capacity and low cost
    cap_per_demand = capacities / (demands.sum() + 1e-9)
    cap_adjusted = open_cost + avg_assign_cost - 10 * cap_per_demand
    by_cap_adj = np.argsort(cap_adjusted)
    for k in [8, 10, 12, 15]:
        if k <= n_fac:
            candidate_subsets.append(set(by_cap_adj[:k].tolist()))
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in candidate_subsets:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Evaluate each subset
    best_result = None
    best_total = float('inf')
    
    # Sort subsets by expected quality (fewer facilities with low open cost first)
    def subset_key(s):
        sc = sum(open_cost[f] for f in s)
        sa = sum(avg_assign_cost[f] for f in s)
        return sc + sa
    
    unique_subsets.sort(key=subset_key)
    
    for subset in unique_subsets:
        if time.time() > deadline - 20:
            break
        
        allowed = np.array(sorted(subset))
        result, assign_cost_total = _multi_start_greedy(demands, capacities, assign_cost, allowed, n_trials=5)
        
        if result is None:
            continue
        
        # Compute total cost
        used_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            used_mask[f] = True
        open_cost_total = open_cost[used_mask].sum()
        total = open_cost_total + assign_cost_total
        
        if total < best_total:
            best_total = total
            best_result = result.copy()
    
    # Local search on best result
    if best_result is not None:
        # Determine which facilities are actually used
        used_facilities = set(best_result.tolist())
        _local_search(best_result, demands, capacities, assign_cost, used_facilities, max_iters=5000, deadline=deadline - 5)
        
        # Recompute cost
        open_cost_total = sum(open_cost[f] for f in used_facilities)
        assign_cost_total = sum(assign_cost[i, best_result[i]] for i in range(n_cust))
        best_total = open_cost_total + assign_cost_total
        
        # Second local search pass with more iterations if time permits
        if time.time() < deadline - 10:
            used_facilities = set(best_result.tolist())
            _local_search(best_result, demands, capacities, assign_cost, used_facilities, max_iters=3000, deadline=deadline - 2)
            
            # Recompute cost again
            open_cost_total = sum(open_cost[f] for f in used_facilities)
            assign_cost_total = sum(assign_cost[i, best_result[i]] for i in range(n_cust))
            best_total = open_cost_total + assign_cost_total
    
    # Fallback: if no valid result, use greedy on all facilities
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
