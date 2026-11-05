import numpy as np
import time

# Global timer
_start_time = time.time()
_TIMEOUT = 135.0 # Safety margin before 150s limit

def _solve_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0:
        return [], 0.0
        
    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]
        
        if wi <= cap_int:
            prev_row = dp[i-1, :]
            take_row = prev_row[:cap_int - wi + 1] + vi
            dp[i, :] = prev_row
            dp[i, wi:] = np.maximum(prev_row[wi:], take_row)
        else:
            dp[i, :] = dp[i-1, :]

    # Backtrack
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            wi = int(round(w[i-1] * scale))
            c -= wi
            if c < 0:
                break

    selected.reverse()
    return selected, dp[n, cap_int]

def _greedy_fill(weights, values, capacity):
    """Greedily fill knapsack by density."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    w = np.array(weights, dtype=np.float64)
    v = np.array(values, dtype=np.float64)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    order = np.argsort(-densities)
    
    total = 0.0
    cap = capacity
    for idx in order:
        if w[idx] <= cap:
            total += v[idx]
            cap -= w[idx]
            if cap <= 0:
                break
    return total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
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
        if len(feasible) == 0: return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

    # Run DP
    try:
        selected_indices, total_value = _solve_dp(w, v, C, scale)
    except Exception:
        # Fallback on error
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0: return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

    if not selected_indices:
        return -1
    
    # IMPROVEMENT: Use lookahead to pick the best item from the optimal set
    sel_w = w[selected_indices]
    sel_v = v[selected_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sel_densities = sel_v / sel_w
    sel_densities = np.nan_to_num(sel_densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Sort by density descending
    density_order = np.argsort(-sel_densities)
    
    # Consider top K candidates for lookahead
    K = min(10, len(selected_indices))
    candidates = [selected_indices[density_order[i]] for i in range(K)]
    
    best_score = -1
    best_pick = candidates[0]
    
    for cand_idx in candidates:
        cand_w = w[cand_idx]
        cand_v = v[cand_idx]
        
        rem_cap = C - cand_w
        if rem_cap < 0:
            continue
        
        # Simulate greedy fill with remaining items from optimal set
        remaining_in_opt = [i for i in selected_indices if i != cand_idx]
        
        if not remaining_in_opt:
            score = cand_v
        else:
            rem_w_list = [w[i] for i in remaining_in_opt]
            rem_v_list = [v[i] for i in remaining_in_opt]
            score = cand_v + _greedy_fill(rem_w_list, rem_v_list, rem_cap)
        
        if score > best_score:
            best_score = score
            best_pick = cand_idx
    
    return int(best_pick)
