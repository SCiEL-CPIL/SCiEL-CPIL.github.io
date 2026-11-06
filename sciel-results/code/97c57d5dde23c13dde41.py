import numpy as np
import time

# Module-level cache for precomputed structures
_cache = {}

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: picks next item using a lookahead-based score
    that approximates the marginal gain in total value, guided by LP relaxation
    of the remaining subproblem after each candidate selection.
    
    Returns index into the remaining arrays, or -1 to stop.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check feasibility: only items that fit in remaining capacity are candidates
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = 15.0  # seconds max for this call (safety margin within 120s total)
    
    # For each feasible candidate, estimate the total value if we pick it now
    # plus the best we can do with remaining capacity (approximated by LP relaxation)
    best_score = -1.0
    best_index = -1
    
    # Pre-sort by value density for LP relaxation approximation
    # LP relaxation: sort by v/w descending, fill greedily (fractional allowed)
    # This gives an upper bound on achievable value
    
    # To speed up, we'll use a simplified scoring:
    # score(i) = value[i] + LP_relaxation(capacity - weight[i], remaining items excluding i)
    
    # For efficiency with up to 100 items, we can afford O(n^2) per call
    # but we need to be careful about the 120s total budget across all calls
    
    # Strategy: For each candidate i, compute:
    # 1. Immediate value: v[i]
    # 2. Approximate future value: LP relaxation of remaining capacity with other items
    #    LP relaxation upper bound: sort remaining items (excluding i) by density,
    #    fill fractionally until capacity is exhausted
    
    # To make this efficient, precompute sorted order by density
    densities = remaining_values / remaining_weights
    sorted_order = np.argsort(-densities)  # descending density
    
    # For each feasible candidate, compute the score
    for idx in feasible_indices:
        if time.time() - t_start > deadline:
            break
            
        wi = remaining_weights[idx]
        vi = remaining_values[idx]
        cap_rem = remaining_capacity - wi
        
        if cap_rem < 0:
            continue
        
        # Compute LP relaxation upper bound for remaining items (excluding idx)
        # with capacity cap_rem
        lp_value = 0.0
        remaining_cap = cap_rem
        
        for j in sorted_order:
            if j == idx:
                continue
            wj = remaining_weights[j]
            vj = remaining_values[j]
            
            if remaining_cap <= 0:
                break
            
            if wj <= remaining_cap:
                lp_value += vj
                remaining_cap -= wj
            else:
                # Fractional fill
                fraction = remaining_cap / wj
                lp_value += vj * fraction
                remaining_cap = 0
                break
        
        total_score = vi + lp_value
        
        if total_score > best_score:
            best_score = total_score
            best_index = idx
    
    if best_index == -1:
        return -1
    
    return int(best_index)
