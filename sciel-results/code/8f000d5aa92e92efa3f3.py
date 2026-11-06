import numpy as np
import time

_START = time.monotonic()
_TIME_LIMIT = 110.0 # Safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Check time
    if time.monotonic() - _START > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    # Feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Pre-compute sorting orders for different heuristics
    # 1. Density (Value/Weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    densities[weights == 0] = np.inf
    sort_by_density = np.argsort(-densities)
    
    # 2. Value (High to Low)
    sort_by_value = np.argsort(-values)
    
    # 3. Weight (Low to High) - to pack more items
    sort_by_weight = np.argsort(weights)

    # Pre-filter indices for each sort order to only include feasible items
    # This helps in the inner loops
    feasible_set = set(feasible_indices)
    
    # For the inner loops, we iterate through the sorted orders and skip infeasible or the candidate itself
    # To speed up, we can create lists of (index, weight, value) for each sort order containing only feasible items
    # But N=100 is small, so filtering on the fly or using masks is fine.
    
    best_value = -1
    best_idx = -1
    
    # Helper function to perform greedy fill given a sorted order and excluded index
    def greedy_fill(rem_cap, sorted_indices, excluded_idx):
        total_val = 0.0
        current_cap = rem_cap
        for idx in sorted_indices:
            if idx == excluded_idx:
                continue
            if idx not in feasible_set:
                continue
            w = weights[idx]
            if w <= current_cap:
                total_val += values[idx]
                current_cap -= w
                if current_cap <= 1e-9:
                    break
        return total_val

    # We will evaluate each feasible candidate
    # To optimize, we can restrict the candidates to a top-K set if N is large, 
    # but N=100 is small enough to check all.
    
    for i in feasible_indices:
        rem_cap_after_i = remaining_capacity - weights[i]
        
        if rem_cap_after_i < 0:
            continue
            
        # Policy 1: Fill with Density Greedy
        val1 = values[i] + greedy_fill(rem_cap_after_i, sort_by_density, i)
        
        # Policy 2: Fill with Value Greedy
        val2 = values[i] + greedy_fill(rem_cap_after_i, sort_by_value, i)
        
        # Policy 3: Fill with Weight Greedy (Small first)
        val3 = values[i] + greedy_fill(rem_cap_after_i, sort_by_weight, i)
        
        # Take the best estimate for this candidate
        best_estimate = max(val1, val2, val3)
        
        if best_estimate > best_value:
            best_value = best_estimate
            best_idx = i
            
    return int(best_idx)
