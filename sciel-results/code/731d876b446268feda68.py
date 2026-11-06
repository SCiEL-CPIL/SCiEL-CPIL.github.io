import numpy as np
import time

def _simulate_greedy_completion(capacity, weights, values, densities, exclude_idx, time_limit=10.0):
    """
    Simulate greedy-by-density completion excluding one item.
    Returns the total value achievable.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    start_time = time.time()
    
    # Use a boolean array to track used items
    used = np.zeros(n, dtype=bool)
    used[exclude_idx] = True
    
    total_value = 0.0
    sim_cap = capacity
    
    # Pre-sort indices by density descending for faster lookup
    sorted_indices = np.argsort(-densities)
    
    while sim_cap > 0:
        if time.time() - start_time > time_limit:
            break
        
        best_j = -1
        for j in sorted_indices:
            if not used[j] and weights[j] <= sim_cap:
                best_j = j
                break
        
        if best_j == -1:
            break
        
        total_value += values[best_j]
        sim_cap -= weights[best_j]
        used[best_j] = True
    
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # Precompute densities
    densities = remaining_values / remaining_weights
    
    # Sort feasible indices by density descending - try highest density items first
    # This helps find good solutions early and allows early termination
    feasible_sorted = feasible_indices[np.argsort(-densities[feasible_indices])]
    
    best_total = -1
    best_idx = -1
    start_time = time.time()
    
    # Limit the number of candidates to evaluate to stay within time budget
    # With 100 items, evaluating all 100 with O(n^2) simulation might be slow
    # Let's evaluate top K candidates first, then expand if time permits
    K = min(20, len(feasible_sorted))
    
    for idx in feasible_sorted[:K]:
        sim_cap = remaining_capacity - remaining_weights[idx]
        sim_value = remaining_values[idx] + _simulate_greedy_completion(
            sim_cap, remaining_weights, remaining_values, densities, idx
        )
        
        if sim_value > best_total:
            best_total = sim_value
            best_idx = idx
        
        # Check time - if we're running out, stop
        if time.time() - start_time > 100.0:
            break
    
    return int(best_idx)
