"""Improved 0/1 knapsack heuristic with value-scaling DP and 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_time():
    if _START_TIME is None:
        return True
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _greedy_fill(cap, weights, values, order, exclude_idx=None):
    """
    Simulate greedy fill following a given order of indices.
    Returns total value achievable.
    """
    total_val = 0.0
    cap_remaining = cap
    for idx in order:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        if cap_remaining <= 0:
            break
        w = weights[idx]
        if w <= cap_remaining:
            cap_remaining -= w
            total_val += values[idx]
    return total_val

def _dp_value(cap, weights, values):
    """
    Value-scaling DP for 0/1 knapsack.
    Returns approximate optimal value.
    """
    n = len(weights)
    if n == 0 or cap <= 0:
        return 0.0
    
    max_val = int(np.ceil(np.sum(values)))
    # Cap DP size to keep it fast
    max_dp_size = 5000
    if max_val > max_dp_size:
        scale = max_dp_size / max_val
        scaled_values = np.floor(values * scale).astype(int)
        max_val = int(np.ceil(np.sum(scaled_values)))
        if max_val == 0:
            return 0.0
        dp = np.zeros(max_val + 1, dtype=np.float64)
        for i in range(n):
            w_i = weights[i]
            v_i = scaled_values[i]
            if v_i == 0:
                continue
            for v in range(max_val, v_i - 1, -1):
                if dp[v - v_i] + w_i <= cap:
                    candidate = dp[v - v_i] + w_i
                    if candidate < dp[v]:
                        dp[v] = candidate
            # Actually, we want to minimize weight for each value
            # Let me redo this properly
        # Redo with proper min-weight DP
        dp = np.full(max_val + 1, np.inf)
        dp[0] = 0.0
        for i in range(n):
            w_i = weights[i]
            v_i = scaled_values[i]
            if v_i == 0:
                continue
            for v in range(max_val, v_i - 1, -1):
                if dp[v - v_i] + w_i < dp[v]:
                    dp[v] = dp[v - v_i] + w_i
        # Find max value achievable
        for v in range(max_val, -1, -1):
            if dp[v] <= cap:
                return v / scale
        return 0.0
    else:
        dp = np.full(max_val + 1, np.inf)
        dp[0] = 0.0
        for i in range(n):
            w_i = weights[i]
            v_i = int(round(values[i]))
            if v_i == 0:
                continue
            for v in range(max_val, v_i - 1, -1):
                if dp[v - v_i] + w_i < dp[v]:
                    dp[v] = dp[v - v_i] + w_i
        for v in range(max_val, -1, -1):
            if dp[v] <= cap:
                return float(v)
        return 0.0

def _two_step_lookahead(cap, weights, values, density_order, top_k=10):
    """
    For each of the top_k candidates (by density), simulate:
    1. Pick candidate i
    2. Pick best j from remaining (by density)
    3. Greedily fill rest by density
    Returns (best_total_value, best_first_index, best_second_index)
    """
    n = len(weights)
    best_total = -1
    best_i = 0
    best_j = -1
    
    # Get top_k candidates by density
    candidates = density_order[:min(top_k, n)]
    
    for i in candidates:
        if not _check_time():
            break
        w_i = weights[i]
        v_i = values[i]
        remaining_cap = cap - w_i
        if remaining_cap <= 0:
            total = v_i
            if total > best_total:
                best_total = total
                best_i = i
                best_j = -1
            continue
        
        # Find best second item j
        best_second_total = -1
        best_second_j = -1
        for j in density_order:
            if j == i:
                continue
            w_j = weights[j]
            if w_j <= remaining_cap:
                # Greedily fill rest after picking i and j
                cap_rem = remaining_cap - w_j
                total = v_i + values[j]
                for k in density_order:
                    if k == i or k == j:
                        continue
                    if cap_rem <= 0:
                        break
                    w_k = weights[k]
                    if w_k <= cap_rem:
                        cap_rem -= w_k
                        total += values[k]
                if total > best_second_total:
                    best_second_total = total
                    best_second_j = j
        
        if best_second_total > best_total:
            best_total = best_second_total
            best_i = i
            best_j = best_second_j
    
    return best_total, best_i, best_j

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = remaining_capacity
    
    # Filter out infeasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Compute density-sorted order
    density = feasible_values / feasible_weights
    density_order = np.argsort(-density)
    
    # Compute value-sorted order
    value_order = np.argsort(-feasible_values)
    
    # Compute weight-ascending order
    weight_order = np.argsort(feasible_weights)
    
    # Strategy 1: Pure density greedy
    density_total = _greedy_fill(cap, feasible_weights, feasible_values, density_order)
    
    # Strategy 2: Pure value greedy
    value_total = _greedy_fill(cap, feasible_weights, feasible_values, value_order)
    
    # Strategy 3: 2-step lookahead with density
    lookahead_total, lookahead_i, lookahead_j = _two_step_lookahead(
        cap, feasible_weights, feasible_values, density_order, top_k=15
    )
    
    # Strategy 4: DP-based estimate to guide selection
    # Use DP to find a good target, then pick items that align with DP solution
    # For efficiency, use a simplified approach:
    # Pick the item that is part of the best DP solution
    
    best_total = max(density_total, value_total, lookahead_total)
    
    if lookahead_total >= best_total - 1e-9:
        # Use the 2-step lookahead result
        return int(feasible_indices[lookahead_i])
    elif density_total >= best_total - 1e-9:
        return int(feasible_indices[density_order[0]])
    else:
        return int(feasible_indices[value_order[0]])
# EVOLVE-BLOCK-END
