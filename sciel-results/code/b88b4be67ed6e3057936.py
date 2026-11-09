"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # Safety margin before 150s

def _greedy_fill(cap, weights, values, sorted_indices, exclude_idx):
    """
    Vectorized greedy fill: iterate through sorted_indices in order,
    skipping exclude_idx, and greedily add items that fit.
    Returns total value of the greedy completion.
    """
    cap = float(cap)
    total_val = 0.0
    n = len(sorted_indices)
    
    # Pre-extract weights and values for the sorted order to avoid repeated indexing
    sw = weights[sorted_indices]
    sv = values[sorted_indices]
    
    # Find position of exclude_idx in sorted_indices for fast skipping
    # We'll just check each one
    for i in range(n):
        idx = sorted_indices[i]
        if idx == exclude_idx:
            continue
        w = sw[i]
        if w <= cap:
            cap -= w
            total_val += sv[i]
            if cap <= 1e-12:
                break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline check
    if time.time() - _START_TIME > _TIME_LIMIT:
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = remaining_values[feasible] / remaining_weights[feasible]
        return int(feasible[np.argmax(densities)])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # 1. Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # 2. Sort feasible items by value/weight density (descending)
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    
    densities = feasible_values / feasible_weights
    # Use a stable sort that breaks ties by higher absolute value
    # Create composite sort key: primarily density, secondarily value
    # We want descending density, and for ties, descending value
    # np.argsort with kind='stable' preserves order of equal keys
    # So we sort by value first (descending), then by density (descending)
    # But argsort only sorts by one key. Let's use a combined approach.
    
    # Alternative: sort by density descending, then for items with same density, 
    # we want higher value first. We can do this by sorting by a tuple.
    # Since numpy doesn't directly support tuple sorting in argsort, 
    # we'll use lexsort or just sort by density and accept that ties are arbitrary.
    # For better tie-breaking, let's add a tiny perturbation based on value.
    
    # Actually, let's just sort by density descending. The lookahead will handle
    # the rest by evaluating downstream outcomes.
    sorted_local_indices = np.argsort(densities)[::-1]
    
    sorted_global_indices = feasible_indices[sorted_local_indices]
    
    # Pre-extract for the sorted order
    sorted_weights = remaining_weights[sorted_global_indices]
    sorted_values = remaining_values[sorted_global_indices]
    
    # 3. Determine K (number of candidates to look ahead)
    # Increased to 40 since simulation is faster
    K = min(len(sorted_global_indices), 40)
    
    best_candidate_idx = -1
    best_total_value = -1.0
    
    # 4. For each of the top K candidates, simulate the rest
    for i in range(K):
        candidate_idx = sorted_global_indices[i]
        
        val_c = remaining_values[candidate_idx]
        w_c = remaining_weights[candidate_idx]
        
        cap_rem = remaining_capacity - w_c
        
        # Simulate greedy completion with remaining items
        sim_val = _greedy_fill(cap_rem, remaining_weights, remaining_values, 
                               sorted_global_indices, candidate_idx)
        
        total_val = val_c + sim_val
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_candidate_idx = candidate_idx

    return int(best_candidate_idx)
# EVOLVE-BLOCK-END
