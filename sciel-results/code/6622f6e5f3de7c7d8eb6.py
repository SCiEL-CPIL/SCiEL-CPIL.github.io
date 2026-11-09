"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = 330.0
_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _check_deadline():
    return _get_elapsed() < _DEADLINE - 5.0  # 5s safety margin


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid strategy:
    1. Compute Expected Improvement (EI)
    2. Compute Probability of Improvement (PI) 
    3. Compute a hybrid score that balances exploration and exploitation
    
    The strategy adapts based on the uncertainty profile:
    - High uncertainty regions get more exploration weight
    - Low uncertainty regions with good predictions get exploitation weight
    """
    n = len(mu)
    
    # Safety check
    if not _check_deadline():
        # Fallback to simple EI if we're close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Adaptive hybrid score
    # The idea: when uncertainty is high relative to potential improvement,
    # favor exploration; when uncertainty is low and improvement is likely,
    # favor exploitation
    
    # Normalize sigma for adaptive weighting
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1e-9
    
    # Relative uncertainty per point
    rel_uncertainty = sigma_safe / sigma_mean
    
    # Clip to reasonable range
    rel_uncertainty = np.clip(rel_uncertainty, 0.1, 10.0)
    
    # Weight: higher when uncertainty is high (exploration)
    # But also consider how likely improvement is
    exploration_weight = 0.3 + 0.7 * np.clip(rel_uncertainty, 0, 2.0) / 2.0
    exploitation_weight = 1.0 - exploration_weight
    
    # Combine EI and PI with adaptive weights
    # Scale PI to be comparable to EI
    pi_scaled = pi * sigma_safe  # Scale probability by uncertainty
    
    score = exploration_weight * ei + exploitation_weight * pi_scaled
    
    # Add a small bonus for points with very low predicted value
    # This helps converge to the true minimum
    low_mu_bonus = np.where(mu < f_best - 0.1 * sigma_safe, 0.1 * sigma_safe, 0.0)
    score += low_mu_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
