import numpy as np
import time

_START = None
_CALL_COUNT = 0
_DEADLINE_MARGIN = 10.0

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _find_optimal_lambda(weights, values, capacity, max_iter=50):
    """
    Find the Lagrangian multiplier lambda that best balances value and weight.
    We want to find lambda such that sum(w_i for v_i - lambda*w_i > 0) ~ capacity.
    If sum > capacity, lambda is too low (we're overweight).
    If sum < capacity, lambda is too high (we're underweight).
    
    We use bisection on lambda.
    Lower bound: 0
    Upper bound: max(values) / min(weights) (rough estimate of max density)
    """
    if len(weights) == 0:
        return 0.0
    
    # Filter to feasible items first for tighter bounds
    feasible_mask = weights <= capacity
    if not feasible_mask.any():
        return 0.0
        
    w = weights[feasible_mask]
    v = values[feasible_mask]
    
    if len(w) == 0:
        return 0.0
        
    # Initial bounds
    # If lambda is very high, no items are selected.
    # If lambda is 0, all items with positive value are selected.
    lo = 0.0
    hi = np.max(v / np.where(w > 0, w, 1.0)) * 1.5
    
    # Bisection
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        # Calculate total weight of items where v_i - mid*w_i > 0
        scores = v - mid * w
        selected_mask = scores > 0
        total_weight = np.sum(w[selected_mask])
        
        if total_weight > capacity:
            # Too much weight, need higher penalty (lambda)
            lo = mid
        else:
            # Too little weight (or exactly right), can lower penalty to include more
            hi = mid
            
        # Early exit if close enough
        if hi - lo < 1e-6:
            break
            
    return (lo + hi) / 2.0

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _CALL_COUNT
    if remaining_capacity is None:
        return -1
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
    
    now = _now()
    elapsed = now - _START
    _CALL_COUNT += 1
    
    # Feasible check
    feasible = weights <= cap
    if not feasible.any():
        return -1
    
    # Safety fallback: if we are running out of time, use simple density
    if elapsed > (150.0 - _DEADLINE_MARGIN):
        with np.errstate(divide='ignore', invalid='ignore'):
            density = np.where(weights > 0, values / weights, 0.0)
        fd = np.where(feasible, density, -np.inf)
        return int(np.argmax(fd))

    # 1. Find optimal lambda (price of weight)
    # We only care about feasible items for the lambda calculation to keep it tight
    # But strictly speaking, infeasible items have negative "reduced cost" anyway if lambda is high enough.
    # Using all items is safer for the global view, but feasible-only is faster.
    # Let's use all items but weight the feasibility in the selection step.
    
    # Optimization: If n is small, just do it.
    # Bisection is O(N * log(1/eps)). With N=100 and 50 iterations, this is 5000 ops. Very fast.
    
    lam = _find_optimal_lambda(weights, values, cap, max_iter=30)
    
    # 2. Calculate Lagrangian scores: v_i - lambda * w_i
    scores = values - lam * weights
    
    # 3. Select best item
    # We must pick from feasible items.
    # Also, ideally we pick an item with positive score (v > lam*w).
    # If all feasible items have negative scores, it means even the best item doesn't justify its weight cost at this lambda.
    # In that case, we should stop? Or pick the least bad one?
    # The task says "return -1 to stop". 
    # However, usually in these heuristics, if cap > 0 and feasible items exist, we should pick one.
    # If the optimal lambda makes all scores negative, it implies the capacity is "expensive".
    # But since we are forced to build a solution, and the evaluator sums values, 
    # picking an item with positive value is always better than stopping, 
    # UNLESS the item has negative value (not possible here usually) or 
    # the "cost" is so high that we should stop.
    # Wait, the Lagrangian score v - lam*w is a heuristic for "net benefit".
    # If max score < 0, it suggests no item is worth taking.
    # But if we return -1, we get 0 additional value.
    # If we pick an item with v > 0, we get v > 0.
    # So, if max(v) > 0, we should probably pick the best feasible item by value, 
    # even if the Lagrangian score is negative?
    # Actually, if lam is optimal, the items with positive score are the "core" items.
    # If no feasible item has a positive score, it means the capacity is nearly full or the remaining items are poor.
    # Let's check: if all feasible scores are <= 0, fall back to max value feasible item.
    
    feas_scores = np.where(feasible, scores, -np.inf)
    best_idx = int(np.argmax(feas_scores))
    best_score = feas_scores[best_idx]
    
    # If the best Lagrangian score is very low/negative, it might be better to just take max value?
    # Or maybe the lambda is just very high because capacity is tiny.
    # Let's trust the Lagrangian unless it's clearly degenerate.
    # However, a common issue: if capacity is very small, lam becomes huge, making all scores negative.
    # In that case, argmax will pick the item with the "least negative" score, 
    # which corresponds to the highest v/w ratio (density).
    # This is a good fallback!
    
    # One edge case: if there are items with value 0, we might pick them. 
    # But values are usually positive in these tasks.
    
    # Additional check: If the best score is negative, but there is an item with high value, 
    # did we mess up lambda?
    # No, if lam is high, it penalizes weight. If cap is small, lam is high.
    # The item with highest density has the highest (v - lam*w) for a given lam?
    # Not necessarily.
    # But picking the argmax of (v - lam*w) is the standard Lagrangian heuristic.
    
    return best_idx
