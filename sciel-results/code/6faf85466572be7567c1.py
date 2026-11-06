"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    start_time = time.time()
    deadline = 100.0 # Safety margin from 120s/150s limit
    
    # Convert to numpy arrays for easier manipulation
    weights = np.array(remaining_weights)
    values = np.array(remaining_values)
    n = len(weights)
    
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter feasible items: weight <= remaining_capacity
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    # Calculate densities for feasible items
    densities = feasible_values / feasible_weights
    
    # Sort feasible items by density descending
    sorted_order = np.argsort(densities)[::-1]
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    sorted_original_indices = feasible_indices[sorted_order]
    
    best_score = -1
    best_index = -1
    
    # For each candidate item, estimate the total value if we pick it
    # plus the fractional knapsack bound of the rest
    for k in range(len(sorted_weights)):
        if time.time() - start_time > deadline:
            break
            
        cand_orig_idx = sorted_original_indices[k]
        cand_w = sorted_weights[k]
        cand_v = sorted_values[k]
        
        remaining_cap = remaining_capacity - cand_w
        
        # Calculate fractional bound for the rest of the items
        # We iterate through the sorted list, skipping the current candidate
        frac_bound = 0.0
        cap_left = remaining_cap
        
        for j in range(len(sorted_weights)):
            if j == k:
                continue
            if cap_left <= 0:
                break
            w_j = sorted_weights[j]
            v_j = sorted_values[j]
            if w_j <= cap_left:
                cap_left -= w_j
                frac_bound += v_j
            else:
                frac_bound += (cap_left / w_j) * v_j
                break
        
        total_score = cand_v + frac_bound
        
        if total_score > best_score:
            best_score = total_score
            best_index = cand_orig_idx
            
    if best_index == -1:
        # Fallback to density if something went wrong or no item was evaluated
        # Just pick the highest density feasible item
        if len(feasible_indices) > 0:
             return int(np.argmax(densities) if len(densities) == len(feasible_indices) else 0) # This logic is tricky, simpler fallback:
             # Actually, the loop above should always run if feasible_indices is not empty.
             # If deadline hit immediately, we might have best_index = -1.
             # Let's just return the first feasible item by density as a safe fallback.
             return int(feasible_indices[np.argmax(densities)])
             
    return int(best_index)
# EVOLVE-BLOCK-END
