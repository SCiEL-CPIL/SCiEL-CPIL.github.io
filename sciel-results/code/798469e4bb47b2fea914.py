"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Check if any item fits
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
        
    w_f = w[feasible_indices]
    v_f = v[feasible_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_f / w_f
        densities = np.where(np.isinf(densities), 1e18, densities)
        densities = np.where(np.isnan(densities), 0, densities)
        
    sorted_order = np.argsort(-densities)
    sorted_feasible_indices = feasible_indices[sorted_order]
    
    # Parameters for lookahead
    K = 20  # Top K first-level candidates
    L = 10  # Top L second-level candidates
    
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit to be safe
    
    best_value = -1
    best_index = -1
    
    # Iterate over top K candidates for the first pick
    for i in range(min(K, len(sorted_feasible_indices))):
        candidate_i = sorted_feasible_indices[i]
        
        # Check time
        if time.time() > deadline:
            break
            
        rem_cap_i = c - w[candidate_i]
        val_i = v[candidate_i]
        
        # If no capacity left, the value is just val_i
        if rem_cap_i <= 1e-9:
            if val_i > best_value:
                best_value = val_i
                best_index = int(candidate_i)
            continue
        
        # Identify candidates for the second pick
        # We need items that fit in rem_cap_i
        # We can just iterate through the sorted list again, skipping i
        # To be efficient, we only look at the top L feasible items for the second pick
        
        best_val_for_i = val_i
        # We will simulate the second pick and then greedy fill
        
        # Create a list of indices for the second level
        # We want the top L items from the remaining set that fit in rem_cap_i
        # Since sorted_feasible_indices is sorted by density, we can just take the first L that fit
        
        second_level_candidates = []
        count = 0
        for j in range(len(sorted_feasible_indices)):
            if j == i:
                continue
            item_idx = sorted_feasible_indices[j]
            if w[item_idx] <= rem_cap_i:
                second_level_candidates.append(item_idx)
                count += 1
                if count >= L:
                    break
            
        if not second_level_candidates:
            # No second item fits, just greedy fill? 
            # Actually, if no second item fits, the greedy fill will pick 0 items.
            # So total value is val_i.
            if val_i > best_value:
                best_value = val_i
                best_index = int(candidate_i)
            continue
        
        # For each second-level candidate, simulate and greedy fill
        for j in range(len(second_level_candidates)):
            candidate_j = second_level_candidates[j]
            
            # Check time
            if time.time() > deadline:
                break
                
            rem_cap_j = rem_cap_i - w[candidate_j]
            val_j = v[candidate_j]
            
            # Greedy fill for the rest
            # We need to pick items from the set excluding i and j
            # To speed up, we can just iterate through the sorted list again
            
            total_val = val_i + val_j
            current_cap = rem_cap_j
            
            # Greedy fill
            # We iterate through sorted_feasible_indices
            # Skip i and j
            # Take if fits
            
            # Optimization: We can break early if current_cap is 0
            for k in range(len(sorted_feasible_indices)):
                if current_cap <= 1e-9:
                    break
                if k == i or k == j:
                    continue
                item_idx = sorted_feasible_indices[k]
                if w[item_idx] <= current_cap:
                    total_val += v[item_idx]
                    current_cap -= w[item_idx]
            
            # 1-Swap Local Search
            # This is O(N^2) but N is small (remaining items)
            # We need to know which items are in the knapsack.
            # Let's reconstruct the knapsack items for the swap.
            # This is expensive. Let's skip local search for now to keep it fast, 
            # or implement a very cheap version.
            # Given the time limit, maybe just greedy is enough with 2-step lookahead.
            
            if total_val > best_val_for_i:
                best_val_for_i = total_val
                
        if best_val_for_i > best_value:
            best_value = best_val_for_i
            best_index = int(candidate_i)
            
    return best_index
# EVOLVE-BLOCK-END
