"""Bounded lookahead 0/1 knapsack heuristic with Lagrangian shadow-price refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0  # safety margin under the 120 s per-eval / 150 s kill

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def _find_lambda(w, v, cap, tol=1e-7, max_iter=30):
    """
    Find the Lagrange multiplier lambda such that the greedy fill
    using reduced costs (v - lambda * w) approximately exhausts capacity.
    Returns (lambda, total_value, total_weight) of the greedy solution at that lambda.
    """
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0, 0.0, 0.0

    # Bounds for lambda:
    # If lambda is very high, we pick nothing (or only items with high v/w)
    # If lambda is very low (0), we pick max value items
    # We want to find lambda where sum(w_i * x_i) ~ cap
    # x_i = 1 if v_i - lambda * w_i > 0 else 0
    # Actually, the standard Lagrangian relaxation for knapsack maximizes sum(v_i x_i) - lambda * sum(w_i x_i)
    # subject to x_i in {0,1}.
    # The optimal lambda for the relaxation is such that the solution to the relaxed problem
    # (which is just picking items with positive reduced cost) has total weight close to cap.
    # However, since the relaxed problem doesn't have a capacity constraint, it will pick ALL items with positive reduced cost.
    # So we need to find lambda such that the total weight of items with (v_i - lambda * w_i) > 0 is close to cap?
    # No, that's not right. The Lagrangian dual function g(lambda) = max sum(v_i x_i - lambda w_i x_i) s.t. x_i in {0,1}
    # = sum max(0, v_i - lambda w_i).
    # The subgradient of g(lambda) is -sum(w_i x_i(lambda)).
    # We want to find lambda such that sum(w_i x_i(lambda)) <= cap.
    # This is equivalent to finding the smallest lambda such that the total weight of items with v_i/w_i > lambda is <= cap?
    # Wait, x_i = 1 if v_i - lambda w_i > 0 => v_i/w_i > lambda.
    # So yes, we are looking for the lambda that is the "marginal density" where the cumulative weight of items with density > lambda is close to cap.
    
    # Let's sort by density descending.
    safe_w = np.maximum(w, 1e-12)
    densities = v / safe_w
    order = np.argsort(-densities)
    w_sorted = w[order]
    v_sorted = v[order]
    d_sorted = densities[order]
    
    # Cumulative weights
    cum_w = np.cumsum(w_sorted)
    
    # Find the index where cum_w exceeds cap
    # If cum_w[-1] <= cap, we can take all items, lambda = 0
    if cum_w[-1] <= cap + 1e-12:
        return 0.0, float(np.sum(v_sorted)), float(cum_w[-1])
    
    # Find k such that cum_w[k-1] <= cap < cum_w[k]
    # The lambda will be between d_sorted[k] and d_sorted[k-1]
    # Specifically, we take the first k items fully, and a fraction of item k?
    # In the Lagrangian relaxation, we don't take fractions. We take items with d > lambda.
    # If we set lambda to d_sorted[k], we take items 0..k-1. Weight is cum_w[k-1].
    # If we set lambda to d_sorted[k-1], we take items 0..k-2. Weight is cum_w[k-2].
    # We want weight ~ cap.
    # Actually, the "shadow price" is the density of the marginal item.
    
    k = np.searchsorted(cum_w, cap, side='left')
    # k is the first index where cum_w[k] >= cap
    # So we take items 0 to k-1 fully.
    # The marginal item is k.
    # The lambda that makes item k indifferent is d_sorted[k].
    # But if we set lambda = d_sorted[k], we take items 0..k-1 (since d_sorted[0..k-1] >= d_sorted[k] is not strictly true if equal, but usually distinct).
    # Let's just return the density of the marginal item as the shadow price.
    # This is a standard heuristic for knapsack.
    
    if k == 0:
        # Even the first item doesn't fit?
        return d_sorted[0], 0.0, 0.0
    
    # The shadow price is the density of the item that would be partially taken in fractional knapsack.
    # That item is at index k in the sorted order (0-indexed).
    # Items 0..k-1 are fully taken. Item k is the marginal one.
    lambda_val = d_sorted[k]
    
    # Calculate total value and weight for the greedy solution at this lambda
    # This solution takes items with d > lambda_val.
    # Due to floating point, we might take items with d >= lambda_val.
    # Let's just compute the value of the first k items.
    total_val = float(np.sum(v_sorted[:k]))
    total_w = float(cum_w[k-1])
    
    return lambda_val, total_val, total_w

def _greedy_complete(w, v, cap, lambda_hint=None):
    """
    Fast greedy completion.
    If lambda_hint is provided, it is used to order items by reduced cost (v - lambda*w).
    Otherwise, it uses standard density (v/w).
    """
    n = len(w)
    if n == 0 or cap <= 0:
        return 0.0

    if lambda_hint is not None and lambda_hint > 0:
        # Sort by reduced cost descending
        reduced_costs = v - lambda_hint * w
        order = np.argsort(-reduced_costs)
    else:
        safe_w = np.maximum(w, 1e-12)
        order = np.argsort(-v / safe_w)
        
    total = 0.0
    c = float(cap)
    for i in order:
        wi = w[i]
        if wi <= c + 1e-12:
            total += v[i]
            c -= wi
            if c <= 1e-9:
                break
    return total

def _score_candidates(w, v, cap, n, lambda_val):
    """Generate diverse candidate indices using multiple scoring families."""
    safe_w = np.maximum(w, 1e-12)
    density = v / safe_w
    rel_w = w / float(cap) if cap > 0 else np.zeros(n)
    
    # Lagrangian reduced cost
    reduced_cost = v - lambda_val * w
    
    scores = np.vstack([
        v,                                        # 1. pure value
        density,                                  # 2. pure density
        reduced_cost,                             # 3. Lagrangian reduced cost
        reduced_cost + 0.1 * v,                   # 4. blend
        v - 0.02 * w,                             # 5. value minus weight penalty
        density * (1.0 + 0.3 * rel_w),            # 6. density boosted by relative size
        v / (w + 0.1 * cap),                      # 7. smoothed density
        reduced_cost - 0.2 * np.abs(rel_w - 0.25),# 8. reduced cost with size-shape
        v * np.exp(-w / (cap + 1e-9)),            # 9. value decaying with weight
        v * (1.0 - rel_w * 0.5),                  # 10. value discounted by fraction of cap
    ])

    cands = set()
    k = min(n, 12)
    for s in scores:
        top = np.argsort(-s)[:k]
        for t in top:
            cands.add(int(t))

    # Ensure key items are included
    cands.add(int(np.argmax(density)))
    cands.add(int(np.argmax(v)))
    if lambda_val > 0:
        cands.add(int(np.argmax(reduced_cost)))

    return list(cands), density, reduced_cost

def _evaluate_candidate(ci, w, v, cap, n, lambda_val):
    """
    Estimate total value if we pick candidate ci now, using greedy completion
    with the current shadow price.
    """
    used_w = float(w[ci])
    used_v = float(v[ci])
    new_cap = cap - used_w
    if new_cap < -1e-9:
        return -np.inf

    # Build arrays excluding candidate
    mask = np.ones(n, dtype=bool)
    mask[ci] = False
    rw = w[mask]
    rv = v[mask]
    rn = len(rw)

    if rn == 0:
        return used_v

    # Use the current lambda for the completion estimate.
    # This is a fast approximation. A more precise approach would re-compute lambda for the new state,
    # but that is expensive. The current lambda is a good proxy for the marginal value of capacity.
    est_completion = _greedy_complete(rw, rv, new_cap, lambda_hint=lambda_val)
    
    return used_v + est_completion

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1

    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1

    # Only consider feasible items
    feasible = w <= cap + 1e-12
    if not np.any(feasible):
        return -1

    # Fast path if near deadline
    if _remaining_budget() < 0.1:
        feasible_v = np.where(feasible, v / np.maximum(w, 1e-12), -np.inf)
        idx = int(np.argmax(feasible_v))
        return idx if feasible[idx] else -1

    # Compute shadow price (Lagrange multiplier)
    lambda_val, _, _ = _find_lambda(w, v, cap)
    
    # Generate diverse candidates
    cands, density, reduced_cost = _score_candidates(w, v, cap, n, lambda_val)

    # Filter to feasible candidates only
    cands = [c for c in cands if feasible[c]]
    if not cands:
        return -1

    # Adaptive candidate limit
    if n > 80:
        sim_limit = 25
    elif n > 40:
        sim_limit = 35
    else:
        sim_limit = 50

    if len(cands) > sim_limit:
        # Sort by reduced cost for pruning
        cands.sort(key=lambda i: -reduced_cost[i])
        cands = cands[:sim_limit]

    # Evaluate each candidate
    best_idx = cands[0]
    best_est = -np.inf
    ests = []

    for ci in cands:
        est = _evaluate_candidate(ci, w, v, cap, n, lambda_val)
        ests.append((est, ci))
        if est > best_est:
            best_est = est
            best_idx = ci

        # Deadline check
        if _remaining_budget() < 0.5:
            break

    # Soft-max tie-breaking
    if len(ests) >= 2:
        ests.sort(key=lambda x: -x[0])
        top_est = ests[0][0]
        # Consider candidates within 1% of the best estimate
        close = [(e, c) for e, c in ests if e >= top_est * 0.99]
        if len(close) > 1:
            # Among close candidates, pick the one with highest reduced cost
            close.sort(key=lambda x: -reduced_cost[x[1]])
            best_idx = close[0][1]
            best_est = close[0][0]

    return int(best_idx)
# EVOLVE-BLOCK-END
