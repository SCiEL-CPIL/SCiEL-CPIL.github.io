import numpy as np
import time

# Global timer to enforce the 150s limit across all calls
_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s kill

def _lagrangian_pick(w_feas, v_feas, C, max_iter=60):
    """
    Use Lagrangian relaxation with a smooth soft-max surrogate to find a good
    lambda, then pick the item that maximizes v_i - lambda * w_i among items
    that fit. Uses log-sum-exp for smoothness and a final polish pass.
    """
    n = len(w_feas)
    if n == 0:
        return -1
    
    feasible = w_feas <= C
    if not np.any(feasible):
        return -1
    
    # Binary search for lambda such that the "effective" total weight of
    # items with positive Lagrangian score is approximately C
    lo = 0.0
    hi = max(v_feas) / min(w_feas) + 1.0 if min(w_feas) > 0 else 1e6
    
    best_lambda = hi / 2.0
    
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        # Compute scores
        scores = v_feas - mid * w_feas
        
        # Use soft-max weighting to get a smooth measure of "how much" of each
        # item would be taken. This avoids the discontinuity of hard threshold.
        # Scale scores for numerical stability
        max_score = np.max(scores)
        if max_score > 0:
            # Soft weights: exp(score / tau) normalized
            tau = 0.1  # Temperature for smoothness
            log_weights = scores / tau - max_score / tau
            # Clip to avoid overflow
            log_weights = np.clip(log_weights, -50, 50)
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)
            
            # Effective weight: sum of weights * w_feas
            effective_w = np.sum(weights * w_feas)
        else:
            effective_w = 0.0
        
        if abs(effective_w - C) < 1e-3:
            best_lambda = mid
            break
        elif effective_w > C:
            lo = mid
        else:
            hi = mid
        
        best_lambda = mid
    
    # Now pick the item that maximizes v_i - lambda * w_i
    scores = v_feas - best_lambda * w_feas
    scores_feas = scores.copy()
    scores_feas[~feasible] = -np.inf
    
    best_idx = int(np.argmax(scores_feas))
    
    # Polish: if the best score is close to zero or negative, try a few
    # alternative lambdas around best_lambda to see if we can find a better pick
    if scores_feas[best_idx] <= 0.01:
        # Try nearby lambdas
        candidates = [best_idx]
        step = (hi - lo) / 4.0 if hi > lo else 0.01
        for offset in [-step, -step/2, step/2, step]:
            lam_try = best_lambda + offset
            if lam_try < 0:
                continue
            scores_try = v_feas - lam_try * w_feas
            scores_try[~feasible] = -np.inf
            idx_try = int(np.argmax(scores_try))
            if scores_try[idx_try] > scores_feas[best_idx]:
                best_idx = idx_try
                scores_feas = scores_try
    
    # Final fallback to density if all scores are very negative
    if scores_feas[best_idx] < -1e-6:
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v_feas / w_feas
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        densities[~feasible] = -np.inf
        best_idx = int(np.argmax(densities))
    
    return best_idx

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a combination of
    Lagrangian relaxation and exact Dynamic Programming.
    Returns the index into remaining_weights/remaining_values, or -1 to stop.
    """
    global _start_time
    
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
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Extract feasible items
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    n_feas = len(w_feas)
    
    if n_feas == 0:
        return -1
    
    # Handle zero-weight items first
    zero_weight_mask = w_feas == 0
    if np.any(zero_weight_mask):
        zero_weight_indices = np.where(zero_weight_mask)[0]
        # Among zero-weight items, pick the one with highest value
        best_zero = zero_weight_indices[np.argmax(v_feas[zero_weight_indices])]
        return int(feasible_indices[best_zero])
    
    # Determine scale for integer weights
    # Use higher precision for better DP accuracy
    scale = 10000
    cap_int = int(round(C * scale))
    
    if cap_int < 1:
        # If capacity is very small, just pick the lightest feasible item
        lightest_idx = np.argmin(w_feas)
        return int(feasible_indices[lightest_idx])
    
    # Check if DP is feasible (memory/time constraints)
    # For n_feas up to 100 and cap_int up to 125000, DP table is 100*125001 = 12.5M floats = 100MB
    # This might be too large. Use Lagrangian as primary if DP is too big.
    dp_size = (n_feas + 1) * (cap_int + 1)
    if dp_size > 5_000_000:  # Limit to ~40MB for safety
        # Use Lagrangian relaxation
        local_idx = _lagrangian_pick(w_feas, v_feas, C)
        if local_idx != -1:
            return int(feasible_indices[local_idx])
        # Fallback to density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v_feas / w_feas
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible_indices[best_local])
    
    w_int = np.round(w_feas * scale).astype(np.int64)
    
    # Ensure no negative weights (shouldn't happen, but safe)
    w_int = np.maximum(w_int, 0)
    
    # DP Table: dp[i][c] = max value using first i items with capacity c
    dp = np.zeros((n_feas + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n_feas + 1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        # Copy previous row
        dp[i, :] = dp[i-1, :]
        
        if wi <= cap_int:
            # Update: dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi) for c >= wi
            if wi > 0:
                take_vals = dp[i-1, :cap_int+1-wi] + vi
                prev_vals = dp[i-1, wi:cap_int+1]
                dp[i, wi:cap_int+1] = np.maximum(prev_vals, take_vals)
            else:
                # wi == 0
                if vi > 0:
                    dp[i, :] += vi
    
    # Backtrack to find one item in the optimal solution
    c = cap_int
    selected_idx = -1
    
    for i in range(n_feas, 0, -1):
        wi = w_int[i-1]
        
        # Check if item i-1 was taken
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected_idx = i - 1
            c -= wi
            break
        
    if selected_idx != -1:
        return int(feasible_indices[selected_idx])
    else:
        # Fallback to Lagrangian
        local_idx = _lagrangian_pick(w_feas, v_feas, C)
        if local_idx != -1:
            return int(feasible_indices[local_idx])
        # Final fallback to density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v_feas / w_feas
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible_indices[best_local])
