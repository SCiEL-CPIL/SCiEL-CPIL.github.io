import numpy as np
from scipy.special import gamma
from scipy.optimize import brentq
import time

# Global state for adaptive parameters
_last_items = []
_beta = 1.0
_lambda = 1.0
_last_update_count = 0
_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def _estimate_weibull(items):
    """Estimate Weibull shape and scale from a list of items."""
    if len(items) < 10:
        return 1.0, 1.0  # Default to Exponential
    
    items = np.array(items)
    mu = np.mean(items)
    var = np.var(items)
    if mu <= 0:
        return 1.0, 1.0
    
    cv2 = var / (mu * mu)
    
    # Function to solve for beta: Gamma(1+2/beta)/Gamma(1+1/beta)^2 - 1 - cv2 = 0
    def f(beta):
        g1 = gamma(1 + 1/beta)
        g2 = gamma(1 + 2/beta)
        return g2 / (g1 * g1) - 1 - cv2
    
    # Search for root in range [0.1, 10]
    try:
        beta = brentq(f, 0.1, 10)
    except ValueError:
        beta = 1.0
        
    lambda_scale = mu / gamma(1 + 1/beta)
    return beta, lambda_scale

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _last_items, _beta, _lambda, _last_update_count, _start_time, _DEADLINE
    
    # Deadline guard: if we're close to timeout, return immediately with simple heuristic
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        remaining = bins - item
        scores = -remaining
        invalid = bins < item
        scores[invalid] = -np.inf
        return scores
    
    # Update parameters periodically
    _last_items.append(item)
    _last_update_count += 1
    
    if _last_update_count % 10 == 0:
        _beta, _lambda = _estimate_weibull(_last_items[-100:])
        # Keep buffer size manageable
        if len(_last_items) > 200:
            _last_items = _last_items[-100:]
            
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins that can fit the item
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    rem_valid = remaining[valid_mask]
    capacity = 100.0
    
    # Item size relative to capacity
    item_frac = item / capacity
    
    # Adaptive strategy based on Weibull shape parameter
    # beta < 1: heavy-tailed (more large items expected) -> favor worst-fit to keep space
    # beta > 1: light-tailed (more small items) -> favor best-fit to pack tight
    # beta = 1: exponential -> balanced approach
    
    # Map beta to a blending weight
    # w = 1.0 means pure worst-fit (maximize remaining)
    # w = 0.0 means pure best-fit (minimize remaining)
    # w = 0.5 means balanced
    
    if _beta < 1.0:
        # Heavy-tailed: prefer worst-fit
        # beta=0.5 -> w=0.8, beta=0.8 -> w=0.6, beta=1.0 -> w=0.4
        w = 0.4 + 0.4 * (1.0 - _beta)
        w = np.clip(w, 0.2, 0.9)
    elif _beta > 1.0:
        # Light-tailed: prefer best-fit
        # beta=1.0 -> w=0.4, beta=1.5 -> w=0.2, beta=2.0 -> w=0.1
        w = 0.4 - 0.2 * (_beta - 1.0)
        w = np.clip(w, 0.05, 0.4)
    else:
        w = 0.4
    
    # Nonlinear scoring: combine best-fit and worst-fit with power-law transform
    # Normalize remaining by capacity for scale invariance
    rem_norm = rem_valid / capacity
    
    # Best-fit component: prefer smaller remaining (tighter fit)
    # Use power law with alpha < 1 to compress large values
    alpha_best = 0.7
    best_fit_score = -(rem_norm ** alpha_best)
    
    # Worst-fit component: prefer larger remaining (keep space for future)
    # Use power law with alpha > 1 to expand large values
    alpha_worst = 1.3
    worst_fit_score = rem_norm ** alpha_worst
    
    # Blend the two components
    # Higher score = better choice
    # best_fit_score is negative (more negative for larger remaining)
    # worst_fit_score is positive (larger for larger remaining)
    # We want to maximize the combined score
    
    # Normalize both to have similar magnitude
    # best_fit_score ranges from -1 (rem=capacity) to 0 (rem=0)
    # worst_fit_score ranges from 0 (rem=0) to ~1.3 (rem=capacity)
    
    combined = w * worst_fit_score + (1 - w) * best_fit_score
    
    # Additional refinement: penalize fragmentation for small items
    # If item is very small, avoid creating tiny unusable fragments
    if item_frac < 0.1:
        # Penalize bins where remaining after placement is very small (< 5% capacity)
        # This avoids wasting a bin on a tiny item that leaves no useful space
        fragment_penalty = 0.1 * np.maximum(0.05 - rem_norm, 0) ** 2
        combined = combined - fragment_penalty
    
    # For large items, strongly prefer tightest fit to minimize waste
    elif item_frac > 0.5:
        # Extra penalty for loose fits
        loose_penalty = 0.05 * np.maximum(rem_norm - 0.3, 0) ** 2
        combined = combined - loose_penalty
    
    scores[valid_mask] = combined
    
    # Final deadline check
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback to simple best-fit if we're running out of time
        scores = -remaining
        invalid = bins < item
        scores[invalid] = -np.inf
    
    return scores
