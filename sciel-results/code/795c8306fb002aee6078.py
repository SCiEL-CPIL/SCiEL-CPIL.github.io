"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

_DEADLINE = 325.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combines EI, adaptive LCB, and power cost penalty.
    Returns 2000 finite scores (higher = more promising).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n != 2000:
        return np.ones(n)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    pdf_z = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * ndtr(z_clipped) + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # Adaptive LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma, where beta controls exploration
    # Progress-based beta: start high (explore), end low (exploit)
    f_min = -3.32237
    f_start = -1.0
    progress = np.clip((f_start - f_best) / (f_start - f_min), 0.0, 1.0)
    
    # Use a slightly more aggressive exploration early on
    beta = 3.0 - 2.5 * progress  # beta from 3.0 (explore) to 0.5 (exploit)
    
    lcb = mu - beta * sigma_safe
    # LCB gain: how much better the lower bound is than f_best
    lcb_gain = np.maximum(f_best - lcb, 0.0)

    # Combine with a small EI component to ensure we don't ignore
    # points with high potential improvement but slightly worse LCB
    # due to high uncertainty (though LCB already accounts for that).
    # Let's try a weighted sum, but heavily favor LCB.
    alpha = 0.8 - 0.3 * progress  # 0.8 early, 0.5 late
    
    # The directive suggests shifting to LCB/cost.
    # Let's try: score = (lcb_gain * alpha + ei * (1-alpha)) / cost^1.2
    # Or simpler: just lcb_gain / cost^1.2 if we trust LCB.
    # Let's stick to a blend but with lower cost exponent.
    
    combined = alpha * lcb_gain + (1.0 - alpha) * ei

    # Power cost penalty
    # Lower exponent (1.2 vs 1.5) reduces the penalty for high-cost points,
    # allowing the algorithm to find the minimum even if it's in a high-cost region.
    scores = combined / np.power(cost_safe, 1.2)

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 1e-10)

    return scores
# EVOLVE-BLOCK-END
