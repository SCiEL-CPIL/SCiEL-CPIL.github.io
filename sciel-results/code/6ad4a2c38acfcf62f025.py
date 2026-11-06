"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Handle empty or zero capacity
    if len(remaining_weights) == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for efficiency
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Time guard setup
    start_time = time.time()
    timeout_limit = 10.0  # Use 10s out of 120s to be safe
    
    # Scaling factor for DP
    # Capacity is ~12.5, so SCALE=1000 gives resolution 0.001
    SCALE = 1000
    
    # Discretize capacity and weights
    # Use ceil for weights to ensure we don't exceed actual capacity
    C_int = int(remaining_capacity * SCALE)
    
    if C_int <= 0:
        return -1

    W_int = np.ceil(weights * SCALE).astype(int)
    
    # Filter items that fit in the remaining capacity
    mask = W_int <= C_int
    if not np.any(mask):
        return -1
        
    feasible_indices = np.where(mask)[0]
    feasible_W = W_int[feasible_indices]
    feasible_V = values[feasible_indices]
    
    # Check time before starting DP
    if time.time() - start_time > timeout_limit:
        # Fallback to density
        with np.errstate(divide='ignore', invalid='ignore'):
            density = feasible_V / feasible_W
            # Replace inf/nan with 0
            density = np.nan_to_num(density, nan=0.0, posinf=1e9, neginf=0.0)
            best_local_idx = int(np.argmax(density))
            return int(feasible_indices[best_local_idx])

    # DP Table: dp[c] = max value achievable with capacity c
    # Initialize with zeros
    dp = np.zeros(C_int + 1, dtype=float)
    
    # Iterate through feasible items
    for i in range(len(feasible_W)):
        w = feasible_W[i]
        v = feasible_V[i]
        
        if w == 0:
            if v > 0:
                dp += v
            continue
            
        # Vectorized 0/1 Knapsack update
        # Update dp[w:] using dp[:-w]
        # dp_new[w:] = max(dp_old[w:], dp_old[:-w] + v)
        # Since we update in place, we must be careful.
        # The slice dp[:-w] is evaluated before the assignment.
        # However, dp[w:] and dp[:-w] overlap.
        # In NumPy, `dp[w:] = np.maximum(dp[w:], dp[:-w] + v)` works correctly
        # because the RHS `dp[:-w] + v` creates a temporary array.
        # Then np.maximum compares the temp array with the current dp[w:].
        # Then assigns to dp[w:].
        # This is the standard vectorized way to do 0/1 knapsack.
        
        dp[w:] = np.maximum(dp[w:], dp[:-w] + v)
        
        # Periodic time check
        if i % 10 == 0 and time.time() - start_time > timeout_limit:
             # Fallback to density for the remaining evaluation
             with np.errstate(divide='ignore', invalid='ignore'):
                density = feasible_V / feasible_W
                density = np.nan_to_num(density, nan=0.0, posinf=1e9, neginf=0.0)
                best_local_idx = int(np.argmax(density))
                return int(feasible_indices[best_local_idx])

    # Evaluate each feasible item to find the best next pick
    best_index = -1
    best_score = -1.0
    
    # We want to maximize: value[i] + dp[remaining_capacity_after_picking_i]
    # remaining_capacity_after_picking_i = C_int - feasible_W[i]
    
    for i in range(len(feasible_indices)):
        orig_idx = feasible_indices[i]
        w = feasible_W[i]
        v = feasible_V[i]
        
        rest_cap = C_int - w
        # dp[rest_cap] is the optimal value for the rest of the items
        # Note: This is an approximation as dp includes item i, but as discussed,
        # it's a strong heuristic.
        est_total = v + dp[rest_cap]
        
        if est_total > best_score:
            best_score = est_total
            best_index = orig_idx
            
    return int(best_index)

# EVOLVE-BLOCK-END
