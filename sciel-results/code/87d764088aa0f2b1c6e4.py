"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter items that are too heavy or have non-positive value
    # We keep original indices
    valid_indices = []
    for i in range(n):
        if remaining_weights[i] <= remaining_capacity and remaining_values[i] > 0:
            valid_indices.append(i)
            
    if not valid_indices:
        return -1
        
    total_value = sum(remaining_values[i] for i in valid_indices)
    if total_value <= 0:
        return -1
        
    # Determine scale for value DP
    # Target DP size ~ 5000 for better resolution
    target_size = 5000
    scale = target_size / total_value
    max_v_scaled = int(round(total_value * scale))
    
    # Safety cap
    if max_v_scaled > 10000:
        max_v_scaled = 10000
        scale = max_v_scaled / total_value
    elif max_v_scaled < 1:
        max_v_scaled = 1
        scale = 1.0 / total_value

    INF = float('inf')
    
    # DP array: min weight to achieve scaled value v
    dp = np.full(max_v_scaled + 1, INF, dtype=np.float64)
    dp[0] = 0.0
    
    # Store backtracking info
    # prev[i][v] stores the scaled value before adding item i
    # If item i was not used, prev[i][v] = v
    # If item i was used, prev[i][v] = v - v_i_scaled
    # We only need this for valid items.
    # Let's map valid_indices to 0..m-1 for easier DP indexing
    m = len(valid_indices)
    
    # Precompute scaled values for valid items
    scaled_values = [int(round(remaining_values[i] * scale)) for i in valid_indices]
    weights = [remaining_weights[i] for i in valid_indices]
    
    # prev is a list of arrays, one for each item step
    # prev[0] is not used, prev[1] corresponds to item 0, etc.
    prev = [None] * (m + 1)
    
    for i in range(m):
        wi = weights[i]
        vi_scaled = scaled_values[i]
        
        if vi_scaled <= 0:
            # Item has negligible scaled value
            prev[i+1] = np.arange(max_v_scaled + 1, dtype=np.int32)
            continue
            
        # Copy current dp to new_dp
        new_dp = dp.copy()
        
        # Update DP
        # We need to update dp[v] = min(dp[v], dp[v-vi] + wi)
        # And record prev[i+1][v]
        
        # Initialize prev row with identity (item not used)
        prev_row = np.arange(max_v_scaled + 1, dtype=np.int32)
        
        # Only update if vi_scaled <= max_v_scaled
        if vi_scaled <= max_v_scaled:
            # Vectorized update is tricky due to dependency (v depends on v-vi)
            # We must iterate downwards or use a temporary array.
            # Since we need to compare dp[v-vi] + wi with dp[v], and dp[v-vi] is from the *previous* state,
            # we can use the old dp array for reading.
            
            # Create a slice of the old dp shifted by vi_scaled
            # source = dp[0 : max_v_scaled - vi_scaled + 1]
            # dest indices = vi_scaled : max_v_scaled + 1
            
            src = dp[:max_v_scaled - vi_scaled + 1]
            dest_idx = np.arange(vi_scaled, max_v_scaled + 1)
            
            # Calculate potential new weights
            # new_weights = src + wi
            # valid_mask = src < INF
            # candidate = src + wi
            # better = candidate < dp[dest_idx]
            
            valid_mask = src < INF
            if np.any(valid_mask):
                candidate = src + wi
                current = dp[dest_idx]
                
                # Find where candidate is better
                better_mask = candidate < current
                
                if np.any(better_mask):
                    # Update new_dp at these positions
                    # new_dp[dest_idx[better_mask]] = candidate[better_mask]
                    
                    # Update prev_row at these positions to point to v - vi_scaled
                    # prev_row[dest_idx[better_mask]] = dest_idx[better_mask] - vi_scaled
                    
                    # We need to update the actual arrays
                    indices_to_update = dest_idx[better_mask]
                    new_dp[indices_to_update] = candidate[better_mask]
                    prev_row[indices_to_update] = indices_to_update - vi_scaled
            
            # For indices where item was NOT used, prev_row remains identity
            # Note: We only updated new_dp where candidate was better.
            # But we must ensure new_dp is consistent.
            # The logic above updates new_dp only where better.
            # Where not better, new_dp keeps old dp values.
            # This is correct.
            
        dp = new_dp
        prev[i+1] = prev_row

    # Find best_v
    # Find largest v such that dp[v] <= remaining_capacity
    feasible = dp <= remaining_capacity
    if not np.any(feasible):
        return -1
        
    best_v = np.max(np.where(feasible)[0])
    
    if best_v == 0:
        return -1
        
    # Backtrack to find all items in the optimal set
    # We want to pick the best item from this set.
    # Let's collect the indices of items in the optimal solution.
    optimal_items = []
    i = m
    v = best_v
    
    while i > 0:
        prev_v = prev[i][v]
        if prev_v != v:
            # Item i-1 was used
            optimal_items.append(i - 1)
            v = prev_v
        else:
            # Item i-1 was not used
            pass
        i -= 1
        
    if not optimal_items:
        return -1
        
    # Select the best item from optimal_items
    # Heuristic: Pick the item with the highest value.
    # Or highest value/weight ratio?
    # High value is usually a good greedy choice.
    best_idx_in_valid = -1
    best_value = -1
    
    for idx in optimal_items:
        orig_idx = valid_indices[idx]
        val = remaining_values[orig_idx]
        if val > best_value:
            best_value = val
            best_idx_in_valid = orig_idx
            
    return best_idx_in_valid

# EVOLVE-BLOCK-END
