# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for time tracking
_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def _get_elapsed():
    """Return elapsed time since module load, initializing on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded():
    """Check if we are close to the 330s wall-clock limit."""
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function.
    
    Uses an adaptive beta based on the current best value f_best.
    When f_best is poor (high value), beta is high (exploration).
    When f_best is good (low value, close to 0), beta is low (exploitation).
    
    Score to maximize: -mu(x) + beta * sigma(x)
    """
    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Deadline guard: if we're close to the limit, use a fast fallback
    if _is_deadline_exceeded():
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
        if not np.any(np.isfinite(ei_scores)):
            return np.zeros_like(mu)
        return ei_scores

    # Calculate adaptive beta
    # f_best is the minimum observed value. Lower is better.
    # We want high exploration when f_best is high (bad).
    # We want low exploration when f_best is low (good).
    # f_best is non-negative for Weierstrass.
    abs_f_best = abs(f_best)
    
    # If f_best is 0, beta should be 0 (pure exploitation, though sigma might still be high)
    # If f_best is large, beta approaches beta_max.
    beta_max = 3.0
    beta = beta_max * (abs_f_best / (1.0 + abs_f_best))
    
    # Add a small floor to beta to ensure some exploration always?
    # If f_best is 0, we are at the minimum. But we might not be exactly at the global min.
    # A small floor like 0.1 might be safe.
    beta = max(beta, 0.1)
    
    # UCB score for minimization (to be maximized)
    # We want to maximize: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all are -inf, return zeros
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
