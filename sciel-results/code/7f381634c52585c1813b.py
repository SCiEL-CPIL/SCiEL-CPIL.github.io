"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Weighted Improvement for minimization.
    
    Strategy:
    - Score = (f_best - mu) * P(f(x) < f_best)
    - This combines the magnitude of potential improvement (exploitation)
      with the probability of improvement (confidence).
    - It is less exploratory than standard EI (which adds a sigma*phi(z) term)
      and more discriminative than pure POI (which is just probability).
    - This aims to capture the benefit of POI's focus on likely improvements
      while adding a gradient towards better (lower) predicted means.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with very large values
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement
    Phi_z = norm.cdf(z_clipped)
    
    # Weighted Improvement: (f_best - mu) * Phi(z)
    # This is positive when mu < f_best, negative otherwise.
    # It prioritizes points that are both likely to improve and have a large gap.
    scores = (f_best - mu) * Phi_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
