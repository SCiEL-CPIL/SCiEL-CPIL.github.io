# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import heapq


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses a combination of:
    1. Greedy initialization with multiple orderings
    2. Local search with 1-move and 2-swap improvements
    3. Facility opening/closing decisions based on cost trade-offs
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit, 35s buffer)
    
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
        ac = sum(assign_cost[i, result[i]] for i in range(n_cust))
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
            # Sort allowed facilities by cost for this customer
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
    
    def _local_search(result, max_iters=200):
        """
        Local search with 1-move and 2-swap improvements.
        Returns improved result and whether it's feasible.
        """
        result = result.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1-move improvements
            for i in range(n_cust):
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                
                # Try moving to each other facility
                for f in range(n_fac):
                    if f == current_f:
                        continue
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        new_cost = assign_cost[i, f]
                        # Check if opening this facility adds cost
                        # and if closing current facility saves cost
                        delta_assign = new_cost - current_cost
                        delta_open = 0
                        if f not in set(result.tolist()):
                            delta_open += open_cost[f]
                        # Check if current facility becomes empty
                        if load[current_f] - demands[i] < 1e-9:
                            delta_open -= open_cost[current_f]
                        
                        if delta_assign + delta_open < -1e-9:
                            load[current_f] -= demands[i]
                            load[f] += demands[i]
                            result[i] = f
                            improved = True
                            break
            
            # 2-swap improvements
            for i in range(n_cust):
                if improved:
                    break
                fi = result[i]
                for j in range(i + 1, n_cust):
                    fj = result[j]
                    if fi == fj:
                        continue
                    
                    # Check feasibility of swap
                    new_load_fi = load[fi] - demands[i] + demands[j]
                    new_load_fj = load[fj] - demands[j] + demands[i]
                    
                    if new_load_fi <= capacities[fi] + 1e-9 and new_load_fj <= capacities[fj] + 1e-9:
                        delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                       (assign_cost[i, fi] + assign_cost[j, fj])
                        
                        delta_open = 0
                        # Check if fi becomes empty
                        if new_load_fi < 1e-9:
                            delta_open -= open_cost[fi]
                        # Check if fj becomes empty
                        if new_load_fj < 1e-9:
                            delta_open -= open_cost[fj]
                        # Check if we need to open a new facility
                        # (not applicable in swap since both facilities are already used)
                        
                        if delta_assign + delta_open < -1e-9:
                            load[fi] = new_load_fi
                            load[fj] = new_load_fj
                            result[i] = fj
                            result[j] = fi
                            improved = True
                            break
        
        return result
    
    def _try_facility_subset(subset, order=None):
        """Try to solve with a specific subset of facilities"""
        if order is None:
            order = np.argsort(-demands)
        
        result = _greedy_init(order, subset)
        if result is None:
            return None, float('inf')
        
        if not _is_feasible(result):
            return None, float('inf')
        
        cost = _total_cost(result)
        return result, cost
    
    # Generate multiple orderings for greedy
    orderings = [
        np.argsort(-demands),  # Largest first
        np.argsort(demands),   # Smallest first
        np.argsort(assign_cost.mean(axis=1)),  # Cheapest to assign first
        np.random.permutation(n_cust),  # Random
        np.random.permutation(n_cust),  # Another random
    ]
    
    # Generate facility subsets to try
    subsets_to_try = [
        list(range(n_fac)),  # All facilities
    ]
    
    # Sort by open cost, try subsets of cheapest
    by_open = np.argsort(open_cost)
    for k in [10, 12, 15, 18]:
        subsets_to_try.append(by_open[:k].tolist())
    
    # Sort by average assignment cost
    avg_assign = assign_cost.mean(axis=0)
    by_assign = np.argsort(avg_assign)
    for k in [10, 12, 15, 18]:
        subsets_to_try.append(by_assign[:k].tolist())
    
    # Combined metric
    combined = open_cost + avg_assign
    by_combined = np.argsort(combined)
    for k in [10, 12, 15, 18]:
        subsets_to_try.append(by_combined[:k].tolist())
    
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in subsets_to_try:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
    
    # Try all combinations of subsets and orderings
    best_result = None
    best_cost = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 10:
            break
        
        for order in orderings:
            if time.time() > deadline - 10:
                break
            
            result, cost = _try_facility_subset(subset, order)
            if result is not None and cost < best_cost:
                best_cost = cost
                best_result = result.copy()
    
    # If we have a feasible solution, improve it with local search
    if best_result is not None:
        if time.time() < deadline - 5:
            improved = _local_search(best_result, max_iters=300)
            if _is_feasible(improved):
                new_cost = _total_cost(improved)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_result = improved
    
    # Fallback: simple greedy if nothing else worked
    if best_result is None:
        # Try all facilities with largest-first ordering
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
