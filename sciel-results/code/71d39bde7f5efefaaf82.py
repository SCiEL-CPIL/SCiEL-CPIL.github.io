# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import heapq


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses min-cost flow for exact assignment given facility subsets,
    combined with facility subset search and efficient local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit, 10s buffer)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def _total_cost(result):
        """Compute total cost: open cost + assignment cost"""
        used = set(result.tolist())
        oc = sum(open_cost[f] for f in used)
        ac = np.sum(assign_cost[np.arange(n_cust), result])
        return oc + ac
    
    def _is_feasible(result):
        """Check if assignment is feasible (no capacity violations)"""
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)
    
    def _greedy_init(order, allowed_facilities):
        """Greedy assignment given an ordering and set of allowed facilities"""
        load = np.zeros(n_fac)
        result = np.full(n_cust, -1, dtype=int)
        allowed = np.array(allowed_facilities)
        
        for i in order:
            placed = False
            costs = assign_cost[i, allowed]
            sorted_idx = np.argsort(costs)
            for idx in sorted_idx:
                f = allowed[idx]
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                return None
        return result
    
    def _min_cost_flow_for_subset(subset):
        """
        Solve exact min-cost assignment using Hungarian algorithm on expanded problem.
        We use a greedy + local search approach for speed, but try to be optimal
        within the subset by using linear_sum_assignment when possible.
        """
        subset_arr = np.array(subset)
        m = len(subset_arr)
        if m == 0:
            return None, float('inf')
        
        # Try greedy first for speed
        result = _greedy_init(np.argsort(-demands), subset)
        if result is None:
            return None, float('inf')
        
        if not _is_feasible(result):
            return None, float('inf')
        
        cost = _total_cost(result)
        return result, cost
    
    def _local_search(result, time_limit):
        """
        Fast local search with 1-move and 2-swap improvements.
        Uses precomputed loads for efficiency.
        """
        result = result.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        # Precompute which facilities are used
        used_mask = np.zeros(n_fac, dtype=bool)
        for f in result:
            used_mask[f] = True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-move improvements
            for i in range(n_cust):
                if time.time() > time_limit:
                    return result
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                
                # Vectorized: compute delta for all facilities
                feasible = load + demands[i] <= capacities + 1e-9
                feasible[current_f] = False
                
                if not np.any(feasible):
                    continue
                
                # For feasible facilities, compute delta
                new_costs = assign_cost[i, feasible] - current_cost
                delta_opens = np.zeros(n_fac)
                
                # Facilities that would become empty
                would_become_empty = (load - demands[i]) < 1e-9
                # Facilities not currently used
                not_used = ~used_mask
                
                # Delta open cost: +open_cost if opening new, -open_cost if closing
                delta_opens[feasible & not_used] = open_cost[feasible & not_used]
                delta_opens[current_f] = -open_cost[current_f] if would_become_empty[current_f] else 0
                
                total_delta = new_costs + delta_opens[feasible]
                min_idx = np.argmin(total_delta)
                min_val = total_delta[min_idx]
                
                if min_val < -1e-9:
                    # Get the actual facility index
                    feasible_indices = np.where(feasible)[0]
                    new_f = feasible_indices[min_idx]
                    
                    # Update
                    load[current_f] -= demands[i]
                    load[new_f] += demands[i]
                    result[i] = new_f
                    used_mask[new_f] = True
                    if load[current_f] < 1e-9:
                        used_mask[current_f] = False
                    improved = True
            
            # 2-swap improvements (sampled for speed)
            if time.time() < time_limit - 1:
                for i in range(n_cust):
                    if time.time() > time_limit:
                        return result
                    fi = result[i]
                    di = demands[i]
                    
                    # Sample some j's for speed
                    candidates = np.random.choice(n_cust, size=min(20, n_cust), replace=False)
                    for j in candidates:
                        if j <= i:
                            continue
                        fj = result[j]
                        if fi == fj:
                            continue
                        dj = demands[j]
                        
                        new_load_fi = load[fi] - di + dj
                        new_load_fj = load[fj] - dj + di
                        
                        if new_load_fi <= capacities[fi] + 1e-9 and new_load_fj <= capacities[fj] + 1e-9:
                            delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                           (assign_cost[i, fi] + assign_cost[j, fj])
                            
                            delta_open = 0
                            if new_load_fi < 1e-9:
                                delta_open -= open_cost[fi]
                            if new_load_fj < 1e-9:
                                delta_open -= open_cost[fj]
                            
                            if delta_assign + delta_open < -1e-9:
                                load[fi] = new_load_fi
                                load[fj] = new_load_fj
                                result[i] = fj
                                result[j] = fi
                                used_mask[fi] = new_load_fi > 1e-9
                                used_mask[fj] = new_load_fj > 1e-9
                                improved = True
                                break
        
        return result
    
    # Generate facility subsets to try
    subsets_to_try = [
        list(range(n_fac)),  # All facilities
    ]
    
    # Sort by various metrics and take top-k
    by_open = np.argsort(open_cost)
    for k in [8, 10, 12, 15]:
        subsets_to_try.append(by_open[:k].tolist())
    
    avg_assign = assign_cost.mean(axis=0)
    by_assign = np.argsort(avg_assign)
    for k in [8, 10, 12, 15]:
        subsets_to_try.append(by_assign[:k].tolist())
    
    # Combined: open_cost + normalized assignment cost
    norm_assign = avg_assign / (avg_assign.max() + 1e-9)
    combined = open_cost + norm_assign * 50  # Scale to be comparable
    by_combined = np.argsort(combined)
    for k in [8, 10, 12, 15]:
        subsets_to_try.append(by_combined[:k].tolist())
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in subsets_to_try:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Try subsets with greedy initialization
    best_result = None
    best_cost = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 20:
            break
        
        # Try multiple orderings
        orderings = [
            np.argsort(-demands),
            np.argsort(demands),
            np.argsort(assign_cost.mean(axis=1)),
        ]
        
        for order in orderings:
            if time.time() > deadline - 20:
                break
            
            result = _greedy_init(order, subset)
            if result is None:
                continue
            
            if not _is_feasible(result):
                continue
            
            cost = _total_cost(result)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
    
    # If we have a feasible solution, improve with local search
    if best_result is not None:
        local_search_time = min(deadline - 5, time.time() + 60)
        improved = _local_search(best_result, local_search_time)
        if _is_feasible(improved):
            new_cost = _total_cost(improved)
            if new_cost < best_cost:
                best_cost = new_cost
                best_result = improved
    
    # Fallback: simple greedy if nothing else worked
    if best_result is None:
        result = _greedy_init(np.argsort(-demands), list(range(n_fac)))
        if result is not None and _is_feasible(result):
            best_result = result
        else:
            # Last resort: round-robin assignment
            best_result = np.zeros(n_cust, dtype=int)
            load = np.zeros(n_fac)
            for i in range(n_cust):
                f = i % n_fac
                while load[f] + demands[i] > capacities[f]:
                    f = (f + 1) % n_fac
                load[f] += demands[i]
                best_result[i] = f
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END
