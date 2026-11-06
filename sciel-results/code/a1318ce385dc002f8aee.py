"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_left():
    if _START_TIME is None:
        return True
    # Safety margin: return before 145s of 150s limit
    return (time.time() - _START_TIME) < 145.0


def _fractional_knapsack_value(capacity, weights, values):
    """Compute optimal fractional knapsack value for given capacity and items."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    # Filter to feasible items (weight <= capacity)
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    w = weights[feasible]
    v = values[feasible]
    # Sort by density descending
    order = np.argsort(-v / w)
    w = w[order]
    v = v[order]
    total = 0.0
    cap = capacity
    for wi, vi in zip(w, v):
        if wi <= cap:
            cap -= wi
            total += vi
        else:
            # Take fractional part
            total += vi * (cap / wi)
            break
    return total


def _dp_lookahead(capacity, weights, values, max_items=15):
    """DP-based lookahead for small number of items. Returns best value achievable."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Filter to feasible items
    feasible_mask = weights <= capacity
    w = weights[feasible_mask]
    v = values[feasible_mask]
    n_feasible = len(w)
    
    if n_feasible == 0:
        return 0.0
    
    # If too many items or capacity too large, fall back to fractional
    if n_feasible > max_items or capacity > 5000:
        return _fractional_knapsack_value(capacity, w, v)
    
    # Use DP with integer weights (assuming weights are integers or can be scaled)
    # First, check if weights are close to integers
    w_int = np.round(w).astype(int)
    if not np.allclose(w, w_int):
        # Scale weights to integers
        scale = 10
        w_int = (w * scale).astype(int)
        cap_int = int(capacity * scale)
    else:
        w_int = w.astype(int)
        cap_int = int(capacity)
    
    if cap_int > 10000:
        return _fractional_knapsack_value(capacity, w, v)
    
    # DP table
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    
    for i in range(n_feasible):
        wi = w_int[i]
        vi = v[i]
        if wi > 0:
            for c in range(cap_int, wi - 1, -1):
                candidate = dp[c - wi] + vi
                if candidate > dp[c]:
                    dp[c] = candidate
    
    return dp[cap_int]


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter to feasible items only
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1

    # If only one feasible item, take it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # One-step lookahead: for each feasible candidate, compute value + optimal remainder
    # Use DP for small instances, fractional for larger ones
    best_val = -1.0
    best_idx = int(feasible_indices[0])

    for idx in feasible_indices:
        w_i = remaining_weights[idx]
        v_i = remaining_values[idx]
        new_cap = remaining_capacity - w_i

        # Build arrays for remaining items excluding idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        w_rest = remaining_weights[mask]
        v_rest = remaining_values[mask]

        # Use DP-based lookahead for better accuracy
        total_val = v_i + _dp_lookahead(new_cap, w_rest, v_rest)

        if total_val > best_val:
            best_val = total_val
            best_idx = int(idx)

        # Time guard
        if not _time_left():
            break

    return best_idx
# EVOLVE-BLOCK-END
