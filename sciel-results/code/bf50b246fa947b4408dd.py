import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1
    
    # Determine scaling factor
    # Assume weights have up to 2-3 decimal places. 
    # Using 1000 is safe for 3 decimal places.
    scale = 1000
    
    # Convert to lists for easy indexing
    weights = list(remaining_weights)
    values = list(remaining_values)
    n = len(weights)
    
    # Scale capacity and weights to integers
    cap = int(remaining_capacity * scale)
    if cap <= 0:
        return -1
        
    # Filter out items with weight > cap (they can't be picked)
    # Also filter out negative weights/values just in case
    valid_indices = []
    valid_weights_int = []
    valid_values = []
    
    for i in range(n):
        w = weights[i]
        v = values[i]
        if w > 0 and v > 0:
            w_int = int(w * scale)
            if w_int <= cap:
                valid_indices.append(i)
                valid_weights_int.append(w_int)
                valid_values.append(v)
                
    if not valid_indices:
        return -1
        
    m = len(valid_indices)
    
    # DP Table: dp[i][c] = max value using first i valid items with capacity c
    # We only need the previous row to compute the next, but we need the whole table for reconstruction?
    # Actually, we can store the table to reconstruct.
    # Size: m x (cap + 1)
    # m <= 100, cap <= 12500
    # 100 * 12500 = 1.25M floats. ~10MB. OK.
    
    dp = np.zeros((m + 1, cap + 1), dtype=np.float64)
    
    # Fill DP table
    for i in range(m):
        w_int = valid_weights_int[i]
        v = valid_values[i]
        # Copy previous row
        dp[i+1, :] = dp[i, :]
        if w_int > 0:
            # Vectorized update
            # dp[i+1, w_int:] = max(dp[i+1, w_int:], dp[i, :-w_int] + v)
            dp[i+1, w_int:] = np.maximum(dp[i+1, w_int:], dp[i, :-w_int] + v)
            
    # Reconstruct solution
    # Start from full capacity
    c = cap
    chosen_idx_in_valid = -1
    
    for i in range(m - 1, -1, -1):
        w_int = valid_weights_int[i]
        v = valid_values[i]
        
        # Check if item i was taken
        # dp[i+1][c] should be dp[i][c-w] + v if taken
        # Note: floating point comparison. Use tolerance?
        # Since we used exact integers for weights and floats for values, 
        # and np.maximum is deterministic, the values should be consistent.
        if c >= w_int:
            val_if_taken = dp[i, c - w_int] + v
            # Check if taking this item leads to the optimal value for current state
            if np.isclose(dp[i+1, c], val_if_taken, rtol=1e-9, atol=1e-9):
                # This item is part of an optimal solution
                # We can pick this one
                chosen_idx_in_valid = i
                break
        # If not taken, we just continue to i-1 with same capacity c
        
    if chosen_idx_in_valid == -1:
        # Fallback: pick first valid item? Or -1?
        # If DP says max value is 0, we should stop.
        if dp[m, cap] <= 0:
            return -1
        # If we couldn't reconstruct (shouldn't happen), pick first
        return valid_indices[0]
        
    return valid_indices[chosen_idx_in_valid]
