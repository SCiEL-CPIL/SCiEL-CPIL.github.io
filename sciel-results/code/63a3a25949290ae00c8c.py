import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a Lagrangian heuristic
    with smoothed scoring and two-step lookahead.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for speed
    w = np.array(remaining_weights, dtype=float)
    v = np.array(remaining_values, dtype=float)
    
    # Filter for feasible items (weight <= capacity)
    feasible_mask = w <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    indices_feas = np.where(feasible_mask)[0]
    
    n_feas = len(w_feas)
    
    # Calculate densities
    densities = v_feas / w_feas
    
    # Sort by density descending to find the fractional knapsack marginal density
    sorted_indices = np.argsort(-densities)
    sorted_w = w_feas[sorted_indices]
    sorted_v = v_feas[sorted_indices]
    sorted_dens = densities[sorted_indices]
    
    # Find the marginal density lambda
    current_weight = 0.0
    lambda_val = 0.0
    
    for i in range(n_feas):
        if current_weight + sorted_w[i] <= remaining_capacity:
            current_weight += sorted_w[i]
            if i == n_feas - 1:
                lambda_val = 0.0
                break
        else:
            lambda_val = sorted_dens[i]
            break
    else:
        lambda_val = 0.0

    # Smoothed scoring: use soft-max to avoid hard max, then pick argmax of smoothed scores
    # Score: v_i - lambda * w_i, but with a smoothing parameter
    base_scores = v_feas - lambda_val * w_feas
    
    # Check if any positive score
    if np.max(base_scores) <= 0:
        return -1
    
    # Two-step lookahead: for each candidate, estimate the value of completing
    # the knapsack after adding that item using a greedy density fill
    best_score = -np.inf
    best_idx = -1
    
    # Limit lookahead to top K candidates by base score for efficiency
    K = min(20, n_feas)
    top_k_indices = np.argsort(-base_scores)[:K]
    
    for idx in top_k_indices:
        if time.time() > deadline:
            break
            
        # Simulate adding this item
        new_capacity = remaining_capacity - w_feas[idx]
        if new_capacity < 0:
            continue
            
        # Greedy fill the remaining capacity with density-sorted items
        # Exclude the chosen item
        remaining_w = np.delete(w_feas, idx)
        remaining_v = np.delete(v_feas, idx)
        
        if len(remaining_w) == 0:
            total_value = v_feas[idx]
        else:
            # Sort by density
            rem_dens = remaining_v / remaining_w
            rem_sorted = np.argsort(-rem_dens)
            rem_w_sorted = remaining_w[rem_sorted]
            rem_v_sorted = remaining_v[rem_sorted]
            
            fill_value = 0.0
            fill_weight = 0.0
            for j in range(len(rem_w_sorted)):
                if fill_weight + rem_w_sorted[j] <= new_capacity:
                    fill_weight += rem_w_sorted[j]
                    fill_value += rem_v_sorted[j]
                else:
                    # Fractional part
                    frac = (new_capacity - fill_weight) / rem_w_sorted[j]
                    if frac > 0:
                        fill_value += frac * rem_v_sorted[j]
                    break
            
            total_value = v_feas[idx] + fill_value
        
        if total_value > best_score:
            best_score = total_value
            best_idx = idx
    
    if best_idx == -1:
        # Fallback to base score
        best_idx_in_feas = np.argmax(base_scores)
    else:
        best_idx_in_feas = best_idx
    
    if base_scores[best_idx_in_feas] <= 0:
        return -1
        
    return int(indices_feas[best_idx_in_feas])
