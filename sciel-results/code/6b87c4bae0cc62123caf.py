"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _find_optimal_lambda(capacity, weights, values, max_iter=60, tol=1e-10):
    """
    Binary search for the Lagrange multiplier lambda such that the
    sum of weights of items with (v_i - lambda*w_i) > 0 is close to capacity.
    
    Returns the lambda value.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    feasible = weights <= capacity + 1e-12
    if not np.any(feasible):
        return 0.0
    
    w = weights[feasible]
    v = values[feasible]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    
    # Filter out any non-finite densities
    valid = np.isfinite(densities) & (densities > 0)
    if not np.any(valid):
        return 0.0
    
    d_valid = densities[valid]
    
    lo = 0.0
    hi = d_valid.max() + 1e-6
    
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        
        # Items with v_i - mid * w_i > 0, i.e., v_i/w_i > mid
        mask = densities > mid
        cum_weight = np.sum(weights[mask])
        
        if cum_weight < capacity:
            lo = mid
        else:
            hi = mid
        
        if hi - lo < tol:
            break
    
    return (lo + hi) / 2.0


def _select_by_lagrangian(capacity, weights, values):
    """
    Select the item that maximizes (v_i - lambda * w_i) where lambda is
    tuned to approximately exhaust capacity. Ties broken by higher value.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return -1
    
    feasible = weights <= capacity + 1e-12
    if not np.any(feasible):
        return -1
    
    # Find the optimal lambda
    lam = _find_optimal_lambda(capacity, weights, values)
    
    # Compute Lagrangian scores: v_i - lambda * w_i
    scores = values - lam * weights
    
    # Only consider feasible items
    scores[~feasible] = -np.inf
    
    # Among items with positive Lagrangian score, pick the one with highest score
    # If all are negative, fall back to highest density
    positive_mask = feasible & (scores > 0)
    
    if np.any(positive_mask):
        # Pick the item with highest (v - lambda*w)
        candidates = np.where(positive_mask)[0]
        best_score = -np.inf
        best_idx = -1
        for idx in candidates:
            s = scores[idx]
            if s > best_score or (abs(s - best_score) < 1e-12 and values[idx] > values[best_idx]):
                best_score = s
                best_idx = idx
        return int(best_idx)
    else:
        # Fallback: highest value/weight density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = values / weights
        densities[~feasible] = -np.inf
        return int(np.argmax(densities))


def _greedy_complete(capacity, weights, values, strategy='density'):
    """
    Complete a knapsack solution greedily.
    strategy: 'density' (v/w), 'value' (v), 'weight' (w, lightest first),
              'lagrangian' (v - lambda*w)
    Returns total value.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    feasible = weights <= capacity + 1e-12
    if not np.any(feasible):
        return 0.0
    
    w = weights.copy()
    v = values.copy()
    cap = capacity
    
    # Build ordering
    if strategy == 'density':
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = v / w
        scores[~feasible] = -np.inf
    elif strategy == 'value':
        scores = v.copy()
        scores[~feasible] = -np.inf
    elif strategy == 'weight':
        scores = -w.copy()  # lighter first
        scores[~feasible] = np.inf
    elif strategy == 'lagrangian':
        lam = _find_optimal_lambda(capacity, weights, values)
        scores = v - lam * w
        scores[~feasible] = -np.inf
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = v / w
        scores[~feasible] = -np.inf
    
    order = np.argsort(-scores)
    
    total_value = 0.0
    total_weight = 0.0
    
    for idx in order:
        if w[idx] <= cap - total_weight + 1e-12:
            total_value += v[idx]
            total_weight += w[idx]
        if total_weight >= cap - 1e-12:
            break
    
    return total_value


def _estimate_with_lookahead(capacity, weights, values, first_idx, max_depth=2):
    """
    Estimate total value if we pick first_idx first, then complete greedily.
    """
    n = len(weights)
    wi = weights[first_idx]
    vi = values[first_idx]
    
    if wi > capacity + 1e-12:
        return 0.0
    
    cap_rem = capacity - wi
    if cap_rem < -1e-12:
        return 0.0
    
    # Remove first_idx
    mask = np.ones(n, dtype=bool)
    mask[first_idx] = False
    rem_w = weights[mask]
    rem_v = values[mask]
    
    # Try multiple completion strategies
    best_val = 0.0
    for strat in ['density', 'lagrangian']:
        val = _greedy_complete(cap_rem, rem_w, rem_v, strategy=strat)
        best_val = max(best_val, val)
    
    return vi + best_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity + 1e-12
    if not np.any(feasible_mask):
        return -1
    
    # Deadline guard
    elapsed = _get_elapsed()
    if elapsed > 130.0:
        # Fallback to simple Lagrangian
        return _select_by_lagrangian(remaining_capacity, remaining_weights, remaining_values)
    
    # Get feasible indices
    feasible_indices = np.where(feasible_mask)[0]
    
    # If few items, evaluate all
    if len(feasible_indices) <= 15:
        candidates = feasible_indices
    else:
        # Generate candidate set
        # 1. Top by Lagrangian score
        lam = _find_optimal_lambda(remaining_capacity, remaining_weights, remaining_values)
        lag_scores = remaining_values - lam * remaining_weights
        lag_scores[~feasible_mask] = -np.inf
        
        # 2. Top by density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = remaining_values / remaining_weights
        densities[~feasible_mask] = -np.inf
        
        # 3. Top by value
        val_scores = remaining_values.copy()
        val_scores[~feasible_mask] = -np.inf
        
        # 4. Lightest items
        weight_scores = -remaining_weights.copy()
        weight_scores[~feasible_mask] = np.inf
        
        K = max(10, n // 4)
        
        top_lag = feasible_indices[np.argsort(-lag_scores[feasible_indices])[:K]]
        top_den = feasible_indices[np.argsort(-densities[feasible_indices])[:K]]
        top_val = feasible_indices[np.argsort(-val_scores[feasible_indices])[:K // 2]]
        top_light = feasible_indices[np.argsort(remaining_weights[feasible_indices])[:K // 3]]
        
        candidates = np.unique(np.concatenate([top_lag, top_den, top_val, top_light]))
    
    # Evaluate each candidate
    best_estimate = -np.inf
    best_pick = -1
    
    for idx in candidates:
        est = _estimate_with_lookahead(remaining_capacity, remaining_weights, remaining_values, idx)
        if est > best_estimate:
            best_estimate = est
            best_pick = int(idx)
    
    if best_pick == -1:
        # Fallback
        return _select_by_lagrangian(remaining_capacity, remaining_weights, remaining_values)
    
    return best_pick
# EVOLVE-BLOCK-END
