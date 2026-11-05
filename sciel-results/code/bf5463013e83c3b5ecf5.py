"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit
_SCALE = 1000

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item using exact Dynamic Programming to pick an item
    from the optimal solution set for the current subproblem.
    Uses vectorized DP updates for speed.
    Falls back to simple density greedy if time is running out or weights are not integer-friendly.
    """
    current_time = time.time()
    if current_time - _start_time > _TIME_LIMIT:
        # Fallback: simple density greedy
        if len(remaining_weights) == 0:
            return -1
        densities = remaining_values / remaining_weights
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Check if weights can be scaled to integers
    try:
        w_scaled = remaining_weights * _SCALE
        w_int = np.round(w_scaled).astype(int)
        if not np.allclose(w_scaled, w_int, atol=1e-9):
            raise ValueError("Weights not integer-friendly")
    except:
        # Fallback to density greedy if DP not applicable
        densities = remaining_values / remaining_weights
        return int(np.argmax(densities))

    c_int = int(round(remaining_capacity * _SCALE))
    if c_int < 0:
        return -1

    # DP arrays
    dp = np.zeros(c_int + 1, dtype=np.float64)
    choice = np.zeros((n, c_int + 1), dtype=np.int8)

    # Fill DP using vectorized updates
    for i in range(n):
        w = w_int[i]
        v = remaining_values[i]
        if w > c_int:
            continue
        
        # Vectorized update:
        # dp[w:] = max(dp[w:], dp[:-w] + v)
        new_vals = dp[:-w] + v
        old_vals = dp[w:]
        
        better_mask = new_vals > old_vals
        idxs = np.where(better_mask)[0]
        
        if len(idxs) > 0:
            # Update dp
            dp[w:][idxs] = new_vals[idxs]
            # Update choice
            choice[i, w:][idxs] = 1
            # Note: choice[i, c] remains 0 where item i is NOT picked

    # Backtrack to find all items in the optimal solution
    optimal_items = []
    c = c_int
    for i in range(n - 1, -1, -1):
        if choice[i, c] == 1:
            optimal_items.append(i)
        c -= w_int[i]
        
    if not optimal_items:
        return -1
        
    # Select the best item from the optimal set
    # Use density as a tie-breaker to potentially leave better options for future steps
    best_idx = -1
    best_density = -1
    for i in optimal_items:
        # Ensure division by zero is handled (though weights > 0 usually)
        if remaining_weights[i] > 0:
            density = remaining_values[i] / remaining_weights[i]
        else:
            density = float('inf')
        if density > best_density:
            best_density = density
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
