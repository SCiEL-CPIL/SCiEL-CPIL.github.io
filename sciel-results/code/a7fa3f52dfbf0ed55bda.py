"""Rewritten acquisition function using a hybrid strategy."""
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # Kill after 330s wall-clock, use 320s as safety margin


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that balances exploration and exploitation.
    Uses a blend of Expected Improvement and a penalty for high uncertainty
    to encourage exploration in early stages, then shifts to exploitation.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Penalize high uncertainty to avoid pure exploration
    # This creates a balance: high EI is good, but very high sigma reduces score
    # The penalty term encourages the algorithm to focus on regions with
    # reasonable confidence when f_best is already low
    
    # Adaptive penalty: as f_best approaches 0, we want more exploitation
    # When f_best is large, we can afford more exploration
    if f_best > 1.0:
        # Early stage: favor exploration more
        exploration_weight = 0.3
    elif f_best > 0.1:
        # Middle stage: balance
        exploration_weight = 0.5
    else:
        # Late stage: favor exploitation
        exploration_weight = 0.7
    
    # Uncertainty penalty: reduces score for high sigma regions
    # Normalize sigma relative to f_best scale
    sigma_scale = max(f_best, 0.01)
    penalty = exploration_weight * sigma / sigma_scale
    
    # Combine: EI minus penalty
    score = ei - penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Small regularization to avoid exact ties
    score += 1e-12 * np.random.randn(len(score))
    
    return score
