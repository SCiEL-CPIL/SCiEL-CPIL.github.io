# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack using vectorized Dynamic Programming.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Global deadline guard
    if not hasattr(select_next_item, '_start_time'):
        select_next_item._start_time = time.time()
    
    elapsed = time.time() - select_next_item._start_time
    if elapsed > 145:  # Safety margin
        return -1

    # Convert to numpy arrays for vectorization
    w_arr = np.array(remaining_weights, dtype=np.float64)
    v_arr = np.array(remaining_values, dtype=np.float64)
    
    # Filter feasible items
    feasible_mask = w_arr <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = w_arr[feasible_mask]
    feasible_values = v_arr[feasible_mask]
    
    m = len(feasible_weights)
    if m == 0:
        return -1
    if m == 1:
        return int(feasible_indices[0])

    # Scale weights and capacity to integers
    scale = 100
    cap_scaled = int(round(remaining_capacity * scale))
    
    # Fallback if capacity is unexpectedly large
    if cap_scaled > 100000:
        densities = feasible_values / feasible_weights
        best_idx = np.argmax(densities)
        return int(feasible_indices[best_idx])

    # Scale weights
    w_scaled = np.round(feasible_weights * scale).astype(int)
    
    # DP Table
    # dp[i][c] = max value using items from i to m-1 with capacity c
    # We only need the previous row to compute the current one, but we need 
    # access to dp[i+1] to check the condition for item i later.
    # So we store all rows.
    # Shape: (m+1) x (cap_scaled + 1)
    # Using float64 for values
    dp = np.zeros((m + 1, cap_scaled + 1), dtype=np.float64)
    
    # Fill DP table backwards
    for i in range(m - 1, -1, -1):
        w_i = w_scaled[i]
        v_i = feasible_values[i]
        
        # Initialize row i with row i+1 (skip item)
        dp[i] = dp[i+1]
        
        if w_i > 0 and w_i <= cap_scaled:
            # Take item: value is v_i + dp[i+1][c - w_i]
            # This corresponds to capacities c from w_i to cap_scaled
            # The source slice is dp[i+1][0 : cap_scaled - w_i + 1]
            # The destination slice is dp[i][w_i : cap_scaled + 1]
            
            # Calculate take values
            take_vals = v_i + dp[i+1][:cap_scaled - w_i + 1]
            
            # Update where taking is better
            # We want max(dp[i+1][w_i:], take_vals)
            current_vals = dp[i+1][w_i:]
            better_vals = np.maximum(current_vals, take_vals)
            dp[i][w_i:] = better_vals
        elif w_i == 0 and v_i > 0:
            # If weight is 0 and value positive, always take it
            # dp[i][c] = v_i + dp[i+1][c]
            dp[i] = v_i + dp[i+1]
        # If w_i == 0 and v_i <= 0, skip (already handled by dp[i] = dp[i+1])
        # If w_i > cap_scaled, skip (already handled by dp[i] = dp[i+1])

    optimal_val = dp[0][cap_scaled]
    
    # Find an item that is part of the optimal solution
    # Condition: v_i + dp[i+1][cap_scaled - w_i] == optimal_val
    # Note: We should check items in the order of the original feasible list (0 to m-1)
    # to maintain consistency with the previous behavior (picking the first valid one).
    
    for i in range(m):
        w_i = w_scaled[i]
        if w_i <= cap_scaled:
            # Calculate value if we pick this item
            # remaining capacity is cap_scaled - w_i
            val_if_pick = feasible_values[i] + dp[i+1][cap_scaled - w_i]
            
            # Use a small tolerance for float comparison
            if abs(val_if_pick - optimal_val) < 1e-9:
                return int(feasible_indices[i])
                
    # Fallback: best density
    densities = feasible_values / feasible_weights
    best_idx = np.argmax(densities)
    return int(feasible_indices[best_idx])

# EVOLVE-BLOCK-END
