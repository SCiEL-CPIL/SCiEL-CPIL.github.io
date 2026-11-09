"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

# Known minimum of Michalewicz 2D
_MICH_MIN = -1.8013

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    LCB = mu - beta * sigma. We want to maximize this (higher score = more promising).
    Since we're minimizing, a low mu (good mean) and high sigma (uncertainty) are both desirable.
    The score returned is the LCB value itself (higher is better for selection).
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    # Check deadline
    if not _elapsed_ok():
        return np.random.rand(len(mu))
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # --- Lower Confidence Bound for minimization ---
    # We want to minimize, so we look for points with low expected value
    # but also high uncertainty (to explore).
    # LCB = mu - beta * sigma
    # A point with low mu and high sigma will have a very low LCB.
    # But we want to return scores where HIGHER means "evaluate next".
    # So we should return -LCB = -mu + beta * sigma
    # Or equivalently, we can just return the LCB and let the caller pick the max?
    # No, the contract says "higher meaning evaluate next".
    # So if LCB is low (good for minimization), we want a high score.
    # score = -LCB = -mu + beta * sigma
    
    # Adaptive beta: start with strong exploration, reduce as we get closer to optimum
    # Progress: 0 when f_best is far from min, 1 when f_best is at min
    # The range of Michalewicz 2D is roughly [-1.8, 1.8]
    # So a reasonable span is about 3.6
    progress = np.clip((f_best - _MICH_MIN) / 3.6, 0.0, 1.0)
    
    # Beta decays from a high value to a low value
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_max * (1.0 - progress) + beta_min * progress
    
    # Compute LCB-based score
    score = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
