import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check for zero weight items with positive value
    zero_weight_mask = (remaining_weights == 0) & (remaining_values > 0)
    if np.any(zero_weight_mask):
        return int(np.where(zero_weight_mask)[0][0])
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    m = len(feasible_indices)
    
    # Calculate densities
    densities = feasible_values / feasible_weights
    
    # Sort indices by density descending
    sorted_order = np.argsort(-densities)
    
    # Determine how many candidates to evaluate based on time budget
    elapsed = time.time() - _start_time
    time_remaining = _TIME_LIMIT - elapsed
    
    # Adaptive candidate count: more time -> more candidates
    if m <= 50:
        num_candidates = m
    elif time_remaining > 10:
        num_candidates = min(m, 30)
    elif time_remaining > 5:
        num_candidates = min(m, 20)
    else:
        num_candidates = min(m, 10)
    
    # Select top candidates by density
    top_k_order = sorted_order[:num_candidates]
    
    # Also include some high-value items that might have lower density
    if m > num_candidates:
        value_sorted = np.argsort(-feasible_values)
        for idx in value_sorted[:5]:
            if idx not in top_k_order:
                top_k_order = np.append(top_k_order, idx)
                if len(top_k_order) >= num_candidates + 5:
                    break
    
    best_score = -1
    best_idx = -1
    
    for k in range(len(top_k_order)):
        if time.time() > _TIME_LIMIT:
            break
            
        idx_in_feasible = feasible_indices[top_k_order[k]]
        w = feasible_weights[top_k_order[k]]
        v = feasible_values[top_k_order[k]]
        
        # Simulate picking this item
        rem_cap = remaining_capacity - w
        sim_val = v
        
        # Greedy fill the rest using density order, skipping the picked item
        for j in range(len(sorted_order)):
            if sorted_order[j] == top_k_order[k]:
                continue
            w_j = feasible_weights[sorted_order[j]]
            if w_j <= rem_cap:
                sim_val += feasible_values[sorted_order[j]]
                rem_cap -= w_j
                if rem_cap <= 0:
                    break
        
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx_in_feasible

    if best_idx == -1:
        # Fallback to highest density item
        best_idx = feasible_indices[sorted_order[0]]
        
    return int(best_idx)
