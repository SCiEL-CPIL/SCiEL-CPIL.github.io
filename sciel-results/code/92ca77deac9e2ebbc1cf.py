import numpy as np
import time

# Deadline guard: track start time globally
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds, with safety margin

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    
    Uses a smooth hybrid strategy:
    - Small items: Best Fit (minimize remaining space) to consolidate.
    - Large items: Worst Fit (maximize remaining space) to avoid fragmentation.
    - Smooth sigmoid transition between regimes centered at 50% of capacity.
    - Variance-based adjustment to favor bins near the mean remaining capacity,
      reducing fragmentation globally.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 2.0:
        # Emergency: return a simple Best Fit score to avoid timeout
        n = len(bins)
        scores = np.full(n, -1e9, dtype=np.float64)
        valid = bins >= item
        if np.any(valid):
            scores[valid] = -bins[valid]
        return scores
    
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = bins[valid_mask]
    
    # Smooth sigmoid transition: t=1 -> small item (Best Fit), t=0 -> large item (Worst Fit)
    # Center at 50 (50% of capacity 100), width 10
    center = 50.0
    width = 10.0
    x = (center - item) / width
    # Numerically stable sigmoid
    if x > 0:
        t = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        t = ex / (1.0 + ex)
    
    # Variance-based adjustment: smooth quadratic penalty on deviation from mean
    # Favor bins whose remaining capacity is close to the mean (reduces fragmentation)
    if n_bins > 1:
        mean_remaining = np.mean(bins)
        # Quadratic penalty: more negative for bins far from mean
        var_term = -(bins - mean_remaining) ** 2
    else:
        var_term = np.zeros(n_bins)
    
    # Weights for variance term in each regime
    # Small items (Best Fit): stronger consolidation incentive
    w_var_small = 0.5
    # Large items (Worst Fit): milder adjustment
    w_var_large = 0.1
    
    # Smoothly blend the variance weight
    w_var = t * w_var_small + (1.0 - t) * w_var_large
    
    # Compute scores for valid bins
    # Best Fit component: -remaining (smaller remaining = higher score)
    # Worst Fit component: +remaining (larger remaining = higher score)
    # Blend: t * (-remaining) + (1-t) * (remaining) = remaining * (1 - 2*t)
    fit_score = valid_remaining * (1.0 - 2.0 * t)
    
    # Add variance adjustment
    scores[valid_mask] = fit_score + w_var * var_term[valid_mask]
    
    return scores
