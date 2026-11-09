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
    Thompson Sampling acquisition function for Bayesian optimization.
    
    For each candidate x, we sample a single value from the posterior:
        s_i = mu_i + sigma_i * z_i,  where z_i ~ N(0, 1)
    
    Then we return scores that are higher for lower sampled values (exploitation)
    and for higher uncertainty (exploration).
    
    The key insight: Thompson Sampling naturally balances exploration and exploitation
    without needing to tune a beta parameter. In early iterations when f_best is poor,
    the high uncertainty dominates. As we find better points, the exploitation signal
    becomes stronger.
    """
    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Deadline guard: if we're close to the limit, use a fast fallback (EI)
    if _is_deadline_exceeded():
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
        if not np.any(np.isfinite(ei_scores)):
            return np.zeros_like(mu)
        return ei_scores
    
    # Thompson Sampling: sample from the posterior for each candidate
    # s_i = mu_i + sigma_i * z_i
    z = np.random.randn(len(mu))
    sampled_values = mu + sigma_safe * z
    
    # We want to maximize a score where:
    # 1. Lower sampled values are better (exploitation)
    # 2. Higher uncertainty is also valuable (exploration)
    # 
    # The score is simply the negative of the sampled value.
    # This naturally balances exploration and exploitation:
    # - When sigma is high, the sampled value has more variance, 
    #   so some samples will be very low (exploration wins)
    # - When sigma is low, the sampled value is close to mu,
    #   so we pick the point with lowest mu (exploitation wins)
    
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all are -inf, return zeros
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
