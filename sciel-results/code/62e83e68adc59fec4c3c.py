"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 115.0  # safety margin before 120s kill


def _check_deadline():
    return (time.time() - _start_time) < _DEADLINE


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add, or -1 to stop.
    
    Uses a combination of:
    1. Value density (v/w) as primary signal
    2. Consideration of remaining capacity vs item weight
    3. A 'regret' or 'opportunity cost' adjustment: prefer items that are
       a good fit for remaining capacity (not too large to leave waste, not too small to be inefficient)
    4. Slight preference for higher absolute value when densities are close
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check deadline
    if not _check_deadline():
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter out items that don't fit
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return -1
    
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    indices_f = np.where(feasible_mask)[0]
    
    n_f = len(w_f)
    if n_f == 0:
        return -1
    
    # If only one feasible item, pick it
    if n_f == 1:
        return int(indices_f[0])
    
    # Compute densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v_f / w_f
    
    # Handle zero-weight items: if weight is 0 and value > 0, it's free value
    zero_weight_mask = w_f == 0
    if np.any(zero_weight_mask):
        # Among zero-weight items, pick the one with highest value
        zw_indices = np.where(zero_weight_mask)[0]
        zw_values = v_f[zw_indices]
        best_zw = zw_indices[np.argmax(zw_values)]
        return int(indices_f[best_zw])
    
    # Replace any remaining inf/nan with 0
    densities = np.where(np.isfinite(densities), densities, 0.0)
    
    # Compute a composite score:
    # 1. Density component (normalized)
    # 2. Fit component: how well the item uses remaining capacity
    #    - Items with weight close to remaining capacity get a boost (good fit)
    #    - But not too close to 0 (wastes capacity on tiny items)
    # 3. Value component: slight boost for higher absolute value
    
    # Normalize density to [0, 1]
    d_min, d_max = np.min(densities), np.max(densities)
    if d_max > d_min:
        d_norm = (densities - d_min) / (d_max - d_min)
    else:
        d_norm = np.ones(n_f)
    
    # Fit score: prefer items whose weight is a reasonable fraction of remaining capacity
    # Use a Gaussian-like bump centered at some fraction of capacity
    # Let's say ideal weight is around 10-50% of remaining capacity
    w_frac = w_f / c  # fraction of remaining capacity
    # Score peaks around 0.15-0.35, decays for very small or very large fractions
    fit_score = np.exp(-((w_frac - 0.25) ** 2) / (2 * 0.15 ** 2))
    
    # Value component: normalize values
    v_min, v_max = np.min(v_f), np.max(v_f)
    if v_max > v_min:
        v_norm = (v_f - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones(n_f)
    
    # Composite score: weighted combination
    # Emphasize density most, then fit, then value
    composite = 0.55 * d_norm + 0.30 * fit_score + 0.15 * v_norm
    
    # Pick the item with highest composite score
    best_local_idx = int(np.argmax(composite))
    return int(indices_f[best_local_idx])
# EVOLVE-BLOCK-END
