import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Filter out items that don't fit individually?
    # The function returns an index into the remaining arrays.
    # If an item doesn't fit, it shouldn't be selected.
    # The loop below handles this.
    
    # Determine scaling factor
    # We want to scale weights to integers for DP.
    # Max capacity is ~12.5.
    # Let's aim for a DP table size of at most 2000-5000 to keep it fast.
    # If min weight is very small, we need higher scale.
    min_w = np.min(weights[weights > 0]) if np.any(weights > 0) else 1.0
    
    # Start with scale 10
    scale = 10
    max_cap_scaled = int(remaining_capacity * scale)
    
    # If max_cap_scaled is too small (< 100), or min_w * scale < 1, adjust scale.
    # Actually, if min_w * scale < 1, we lose precision.
    # Let's ensure min_w * scale >= 1.
    if min_w > 0:
        required_scale = 1.0 / min_w
        if required_scale > scale:
            scale = int(required_scale) + 1
            
    # Cap the scale to avoid huge arrays if weights are tiny (e.g. 1e-6).
    # If scale is huge, DP is too slow.
    # If scale > 1000, fall back to greedy?
    # For now, let's cap at 500.
    if scale > 500:
        scale = 500
        
    max_cap_scaled = int(remaining_capacity * scale)
    
    best_val = -1.0
    best_idx = -1
    
    # Pre-compute scaled weights for all items
    # w_scaled[i] = int(weights[i] * scale)
    # If w_scaled[i] == 0, it's effectively free.
    
    # To speed up, we can pre-compute the list of items and their scaled weights.
    # But we need to exclude candidate i.
    
    # We will iterate over all candidates i.
    for i in range(n):
        w_i = weights[i]
        if w_i > remaining_capacity:
            continue
            
        v_i = values[i]
        rem_cap = remaining_capacity - w_i
        rem_cap_scaled = int(rem_cap * scale)
        
        # If rem_cap is very small, DP is cheap.
        # Initialize DP
        dp = np.zeros(rem_cap_scaled + 1, dtype=float)
        
        # Iterate over other items
        # To avoid O(n^2) overhead in Python loop, we can vectorize the DP update.
        # But the DP update itself is sequential over items.
        # dp[k] = max(dp[k], dp[k-w_j] + v_j)
        
        # We can skip items with w_j > rem_cap
        # And items with w_j_scaled == 0 (add value directly?)
        
        # Let's collect items j != i
        # For performance, we can just loop j in range(n) and skip i.
        
        # Optimization: If rem_cap_scaled is 0, no other items fit.
        if rem_cap_scaled > 0:
            for j in range(n):
                if j == i:
                    continue
                w_j = weights[j]
                if w_j > rem_cap:
                    continue
                w_js = int(w_j * scale)
                if w_js == 0:
                    # Item has negligible weight. Add to all?
                    # Actually, if w_js==0, it fits in any capacity.
                    # We can just add its value to the base?
                    # No, it's 0/1.
                    # If w_js==0, dp[k] = dp[k] + v_j is not correct for 0/1 knapsack DP structure easily in vectorized form without tracking.
                    # However, if w_js==0, it's always beneficial to take it.
                    # We can pre-sum all zero-weight items?
                    # Let's handle w_js==0 by adding to a separate accumulator?
                    # No, simpler: if w_js==0, treat as w_js=1? No, that changes capacity.
                    # If w_js==0, we can just add v_j to all dp entries? 
                    # Wait, if w_js==0, we can take it regardless of capacity.
                    # So we can just add v_j to the total value at the end?
                    # Yes, if weight is 0 (or scales to 0), it doesn't consume capacity.
                    # So we can sum all such values and add to final result.
                    # But we must not double count if we iterate.
                    # Let's collect zero-weight items separately.
                    pass # We'll handle this differently.
                    
                else:
                    # Vectorized DP update
                    # dp[w_js:] = np.maximum(dp[w_js:], dp[:-w_js] + values[j])
                    # This is O(rem_cap_scaled)
                    if w_js <= rem_cap_scaled:
                         dp[w_js:] = np.maximum(dp[w_js:], dp[:-w_js] + values[j])
        
        sim_val = v_i + dp[rem_cap_scaled]
        
        # Handle zero-weight items (scaled to 0)
        # We skipped them in the DP loop above.
        # We should add their values to sim_val.
        # But we need to know which ones are zero-weight.
        # This requires checking w_js == 0 for all j != i.
        # This is O(n) per candidate, so O(n^2) total. Acceptable.
        
        zero_w_val = 0.0
        for j in range(n):
            if j == i:
                continue
            w_j = weights[j]
            if w_j > rem_cap:
                continue
            if int(w_j * scale) == 0:
                zero_w_val += values[j]
                
        sim_val += zero_w_val

        if sim_val > best_val:
            best_val = sim_val
            best_idx = i
            
    return int(best_idx)
