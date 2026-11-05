"""Combined program: Lagrangian relaxation + DP refinement."""
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIMEOUT = 145.0  # Safety margin before 150s limit

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

    # Use 1D DP for memory efficiency and speed
    # dp[c] = max value achievable with capacity c
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    
    # We need to track choices for backtracking, which is expensive.
    # Instead, let's just get the optimal value.
    # But we need the *set* of items to pick one from.
    # So we need backtracking.
    # For N=100, C=1250 (scale=100), a 2D table is 100*1251 ~ 125k floats.
    # This is small. Let's use 2D for simplicity of backtracking.
    
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]
        
        if wi <= cap_int:
            # Copy previous row
            dp[i, :] = dp[i-1, :]
            # Update where c >= wi
            # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            # Vectorized update
            # c ranges from wi to cap_int
            # dp[i, wi:cap_int+1] = max(dp[i-1, wi:cap_int+1], dp[i-1, 0:cap_int-wi+1] + vi)
            
            # Note: dp[i-1, 0:cap_int-wi+1] corresponds to c-wi for c in [wi, cap_int]
            take_vals = dp[i-1, 0:cap_int-wi+1] + vi
            prev_vals = dp[i-1, wi:cap_int+1]
            
            dp[i, wi:cap_int+1] = np.maximum(prev_vals, take_vals)
        else:
            # Item too heavy, copy previous
            dp[i, :] = dp[i-1, :]

    # Backtrack to find selected items
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        # Check if item i-1 was taken
        # If dp[i, c] > dp[i-1, c], it was taken (approximately)
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            wi = int(round(w[i-1] * scale))
            c -= wi
            if c < 0:
                break
        # else: not taken

    selected.reverse()
    return selected, dp[n, cap_int]

def _lagrangian_pick(weights, values, capacity, lambda_val):
    """
    Picks the item maximizing v_i - lambda * w_i among feasible items.
    """
    n = len(weights)
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
    
    # Calculate scores
    scores = np.full(n, -np.inf)
    scores[feasible_mask] = values[feasible_mask] - lambda_val * weights[feasible_mask]
    
    best_idx = np.argmax(scores)
    if scores[best_idx] < 0:
        # If all scores are negative, it might be better to stop?
        # But the problem says return -1 to stop.
        # However, if we must pick, we pick the best.
        # The caller decides if -1 is appropriate.
        pass
        
    return int(best_idx)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
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
    
    # Strategy:
    # 1. Use Lagrangian relaxation to find a good candidate.
    #    We binary search for lambda such that the sum of weights of items with v_i - lambda*w_i > 0
    #    is close to C.
    #    Actually, the standard Lagrangian relaxation for knapsack involves finding lambda
    #    such that the "relaxed" solution (take all items with positive reduced cost)
    #    has total weight close to C.
    #    
    #    Let's find lambda.
    #    Reduced cost r_i = v_i - lambda * w_i.
    #    We take items with r_i > 0.
    #    Let S(lambda) = sum(w_i for i where r_i > 0).
    #    We want S(lambda) ~= C.
    #    S(lambda) is monotonically decreasing with lambda.
    
    # Binary search for lambda
    # Lower bound: 0
    # Upper bound: max(v_i / w_i) + epsilon (so that all r_i < 0, S=0)
    # Or simply max(v_i) / min(w_i) if w_i > 0.
    
    min_w = np.min(w[feasible_mask])
    max_v = np.max(v[feasible_mask])
    
    if min_w <= 0:
        # Handle zero weights separately?
        # If w=0 and v>0, always take it.
        # Let's assume w > 0 for simplicity in binary search bounds, or handle edge cases.
        # For now, if min_w == 0, we might have issues with division.
        # Let's just use a large upper bound.
        lambda_high = max_v * 10 + 1.0
    else:
        lambda_high = max_v / min_w + 1.0
        
    lambda_low = 0.0
    
    # If even at lambda=0, sum of weights < C, then we can take all positive value items.
    # But we are constrained by C.
    # If sum(w_i for v_i > 0) <= C, we take all.
    # But we only pick ONE item per call.
    # The Lagrangian heuristic helps pick the *best* item to add next.
    
    # Let's find lambda such that the "greedy by reduced cost" solution fills capacity.
    # Actually, a simpler heuristic:
    # Pick the item that maximizes v_i - lambda * w_i.
    # If we tune lambda to match the "break-even" density of the marginal item.
    
    # Let's try a few lambda values and see which one gives a "good" pick.
    # Or, use the lambda that makes the sum of weights of items with positive reduced cost
    # closest to C.
    
    best_lambda = 0.0
    min_diff = float('inf')
    
    # Binary search for 20 iterations
    for _ in range(20):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        reduced_costs = v - lambda_mid * w
        # Take items with positive reduced cost AND feasible weight (though if w > C, it's not feasible anyway)
        # But in the relaxation, we ignore the single-item weight constraint?
        # No, the relaxation usually ignores the capacity constraint, replacing it with the penalty.
        # So we take all items with v_i - lambda*w_i > 0.
        # But we must also respect w_i <= C?
        # In the relaxed problem, we can take an item even if w_i > C? No, that's physically impossible.
        # The Lagrangian relaxation of the knapsack problem:
        # max sum(v_i x_i)
        # s.t. sum(w_i x_i) <= C, x_i in {0,1}
        # L(lambda) = sum(v_i x_i) + lambda * (C - sum(w_i x_i))
        # = lambda * C + sum((v_i - lambda*w_i) x_i)
        # To maximize L, we set x_i = 1 if (v_i - lambda*w_i) > 0, else 0.
        # The resulting set S(lambda) might violate sum(w_i) <= C.
        # We want lambda such that sum(w_i for i in S(lambda)) is close to C.
        
        mask_pos = (reduced_costs > 0) & (w <= C) # Only consider feasible items?
        # Actually, if w_i > C, it can never be in a feasible solution, so we can exclude it.
        
        total_w = np.sum(w[mask_pos])
        
        diff = abs(total_w - C)
        if diff < min_diff:
            min_diff = diff
            best_lambda = lambda_mid
            
        if total_w > C:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid
            
    # Now pick the item with max (v_i - best_lambda * w_i)
    # But wait, if we pick this item, do we guarantee a good solution?
    # This is a heuristic.
    
    # Alternative: Use the DP result from inspiration 1.
    # DP is exact. Why not just use DP?
    # The issue with DP is scaling.
    # If we use DP, we get the *optimal* set of items for the *current* state.
    # Then we just need to pick one item from that set.
    # Which one?
    # If we pick any item from the optimal set, the remaining problem will have an optimal solution
    # that is a subset of the original optimal set (minus the picked item).
    # So picking *any* item from the optimal set is a valid move towards the optimal solution.
    # However, the order might matter if there are ties or if the evaluator has some other logic?
    # No, the evaluator just adds the value.
    # So, if we use DP to find the optimal set S*, and pick *any* i in S*,
    # then the total value will be optimal, provided we always pick from the new optimal set in subsequent steps.
    # Since we re-run DP at each step, we will always pick from the current optimal set.
    # Thus, the total value achieved will be the optimal value for the initial instance.
    
    # Why didn't we just do this before?
    # Because DP might be slow?
    # N=100, C=12.5. Scale=100 -> C_int=1250.
    # DP table 100 * 1251 = 125,100 floats.
    # Time: 100 * 1250 = 125,000 operations.
    # This is very fast.
    # 8 instances * 100 steps = 800 calls.
    # 800 * 125,000 = 100,000,000 operations.
    # In Python, this might be ~1-2 seconds.
    # This is well within the 150s limit.
    
    # So, let's just use DP.
    # It is the most robust "structural" change.
    # It guarantees optimality (up to scaling precision).
    
    scale = 100
    
    # Check if capacity is reasonable for DP
    cap_int = int(round(C * scale))
    if cap_int > 50000: 
        # Fallback to heuristic if capacity is huge (unlikely given C=12.5)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0: return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

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
        
    # Pick the first item in the optimal set?
    # Or the one with highest density?
    # It doesn't matter for the total value, as long as we pick from the optimal set.
    # However, picking a "small" item might leave more capacity for later, but since we re-solve DP,
    # the future decisions will adapt.
    # Let's just pick the first one.
    # Actually, let's pick the one with the highest value? Or density?
    # It shouldn't matter. Let's pick the one with index 0 in the sorted list of selected items?
    # The selected_indices list is in increasing index order (from backtracking).
    # Let's just pick the first one.
    
    return int(selected_indices[0])

# EVOLVE-BLOCK-END
