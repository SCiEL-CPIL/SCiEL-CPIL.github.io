"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit
_SCALE = 1000

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item using exact Dynamic Programming to pick an item
    from the optimal solution set for the current subproblem.
    Falls back to simple density greedy if time is running out.
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
    # If not, fallback to heuristic
    try:
        w_scaled = remaining_weights * _SCALE
        w_int = np.round(w_scaled).astype(int)
        if not np.allclose(w_scaled, w_int, atol=1e-9):
            raise ValueError("Weights not integer-friendly")
    except:
        # Fallback to old heuristic if DP not applicable
        densities = remaining_values / remaining_weights
        sorted_indices = np.argsort(-densities)
        best_value = -1
        best_idx = -1
        for cand in np.where(feasible_mask)[0]:
            cur_cap = remaining_capacity - remaining_weights[cand]
            total_val = remaining_values[cand]
            for idx in sorted_indices:
                if idx == cand:
                    continue
                if remaining_weights[idx] <= cur_cap:
                    cur_cap -= remaining_weights[idx]
                    total_val += remaining_values[idx]
            if total_val > best_value:
                best_value = total_val
                best_idx = cand
        return int(best_idx)

    c_int = int(round(remaining_capacity * _SCALE))
    if c_int < 0:
        return -1

    # DP arrays
    # dp[c] = max value with capacity c
    dp = np.zeros(c_int + 1, dtype=np.float64)
    # choice[i][c] = 1 if item i is picked in optimal solution for capacity c
    # We only need to store this for backtracking.
    # Memory: n * (c_int + 1) booleans.
    # 100 * 12500 = 1.25M bytes. OK.
    choice = np.zeros((n, c_int + 1), dtype=np.int8)

    # Fill DP
    for i in range(n):
        w = w_int[i]
        v = remaining_values[i]
        if w > c_int:
            continue
        # Update dp in reverse order
        # dp[c] = max(dp[c], dp[c-w] + v)
        # We use a loop here. NumPy vectorization is hard due to dependency on c-w.
        # However, 1.25M iterations is fast.
        for c in range(c_int, w - 1, -1):
            val = dp[c - w] + v
            if val > dp[c]:
                dp[c] = val
                choice[i][c] = 1
            else:
                choice[i][c] = 0

    # Backtrack to find an item in the optimal solution
    c = c_int
    for i in range(n - 1, -1, -1):
        if choice[i][c] == 1:
            return i
        c -= w_int[i]
        
    # If no item found, return -1
    return -1
# EVOLVE-BLOCK-END
