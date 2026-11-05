"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 135.0  # Safety margin before 150s limit

def _solve_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0 or C <= 0:
        return [], 0.0

    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    # dp[i][c] = max value using first i items with capacity c
    # Using numpy for vectorized row updates
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)

    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]

        if wi <= cap_int:
            prev_row = dp[i-1, :]
            # Copy previous row as base (not taking item)
            dp[i, :] = prev_row
            # Update for taking item: dp[i, wi:] = max(dp[i-1, wi:], dp[i-1, :cap_int-wi+1] + vi)
            take_row = prev_row[:cap_int - wi + 1] + vi
            dp[i, wi:] = np.maximum(prev_row[wi:], take_row)
        else:
            # Item too heavy, copy previous row
            dp[i, :] = dp[i-1, :]

    # Backtrack to find selected items
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        wi = int(round(w[i-1] * scale))
        # Check if item i-1 was taken
        # Use small tolerance for floating point comparison
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            c -= wi
            if c < 0:
                break

    selected.reverse()
    return selected, dp[n, cap_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.

    Strategy: Solve the exact knapsack DP for the current state.
    From the optimal set of items, pick the one with highest density
    (or highest value) to add first. This ensures the sequence of picks
    leads to the globally optimal solution for the current subproblem.
    """
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights, dtype=np.float64)
        v = np.array(remaining_values, dtype=np.float64)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity

    # Scale factor
    scale = 100

    # Check if capacity is reasonable for DP
    cap_int = int(round(C * scale))
    if cap_int > 50000:
        # Fallback to heuristic if capacity is huge
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0:
            return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

    # Run DP to find the optimal set of items
    try:
        selected_indices, total_value = _solve_dp(w, v, C, scale)
    except Exception:
        # Fallback on error
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0:
            return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

    if not selected_indices:
        return -1

    # From the optimal set, pick the item with the highest density
    # This heuristic helps in the sequential construction phase.
    # If we always pick from the global optimal set, we will eventually
    # construct that set (or a close approximation).
    sel_w = w[selected_indices]
    sel_v = v[selected_indices]
    with np.errstate(divide='ignore', invalid='ignore'):
        sel_densities = sel_v / sel_w
    sel_densities = np.nan_to_num(sel_densities, nan=0.0, posinf=1e9, neginf=-1e9)

    best_pick_rel = np.argmax(sel_densities)
    best_idx = selected_indices[best_pick_rel]

    return int(best_idx)
# EVOLVE-BLOCK-END
