"""Improved 0/1 Knapsack heuristic: 1-step lookahead with DP polish for small capacities."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _dp_max_value(weights, values, capacity, scale=1000):
    """Solve 0/1 knapsack via DP on scaled integer weights. Returns max value."""
    if len(weights) == 0 or capacity <= 0:
        return 0.0
    # Scale weights to integers
    w_int = np.maximum(1, np.round(weights * scale)).astype(np.int64)
    cap_int = int(round(capacity * scale))
    n = len(w_int)
    # Filter out items that individually exceed capacity
    valid = w_int <= cap_int
    if not np.any(valid):
        return 0.0
    w_int = w_int[valid]
    v_valid = values[valid]
    n = len(w_int)
    # DP array
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    for i in range(n):
        wi = w_int[i]
        vi = v_valid[i]
        # Update in reverse to avoid using same item twice
        for c in range(cap_int, wi - 1, -1):
            candidate = dp[c - wi] + vi
            if candidate > dp[c]:
                dp[c] = candidate
    return dp[cap_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Deadline check
    if time.time() - _start_time > TIME_LIMIT:
        weights = np.asarray(remaining_weights)
        values = np.asarray(remaining_values)
        if len(weights) == 0:
            return -1
        feasible = weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        feasible_indices = np.where(feasible)[0]
        feasible_weights = weights[feasible_indices]
        feasible_values = values[feasible_indices]
        densities = feasible_values / feasible_weights
        return int(feasible_indices[np.argmax(densities)])

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    n = len(weights)
    
    if n == 0:
        return -1
        
    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    m = len(feasible_weights)
    
    # If only one feasible item, return it
    if m == 1:
        return int(feasible_indices[0])
        
    # Sort feasible items by density descending
    densities = feasible_values / feasible_weights
    sorted_order = np.argsort(-densities)
    
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    sorted_orig_indices = feasible_indices[sorted_order]
    
    best_score = -1.0
    best_item_index = -1
    
    # For each candidate item, simulate picking it and then optimally filling the rest
    for i in range(m):
        w_i = sorted_weights[i]
        v_i = sorted_values[i]
        
        rem_cap = remaining_capacity - w_i
        if rem_cap < 0:
            continue
        total_val = v_i
        
        # Build list of remaining items (excluding candidate i)
        rem_weights = np.concatenate([sorted_weights[:i], sorted_weights[i+1:]])
        rem_values = np.concatenate([sorted_values[:i], sorted_values[i+1:]])
        
        # Use DP for the subproblem if capacity is small enough
        # Scale: with capacity 12.5 and scale 1000, cap_int = 12500
        # DP cost is O(n * cap_int) = O(100 * 12500) = 1.25M operations per candidate
        # With m=100 candidates, total = 125M operations — may be too slow
        # Use DP only when rem_cap is small relative to original, or use greedy for large rem_cap
        
        if rem_cap <= 2.0:
            # Small remaining capacity: use DP for accuracy
            dp_val = _dp_max_value(rem_weights, rem_values, rem_cap, scale=1000)
            total_val += dp_val
        elif rem_cap <= 5.0:
            # Medium: use DP with lower scale for speed
            dp_val = _dp_max_value(rem_weights, rem_values, rem_cap, scale=500)
            total_val += dp_val
        else:
            # Large: use greedy fill (density order)
            rem_cap_local = rem_cap
            for j in range(len(rem_weights)):
                if rem_weights[j] <= rem_cap_local:
                    rem_cap_local -= rem_weights[j]
                    total_val += rem_values[j]
        
        if total_val > best_score:
            best_score = total_val
            best_item_index = int(sorted_orig_indices[i])
            
    return best_item_index
# EVOLVE-BLOCK-END
