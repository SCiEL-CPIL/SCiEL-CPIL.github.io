"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 280  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Thompson Sampling approach (stochastic) to balance exploration and exploitation,
    with cost-awareness via instantaneous improvement per cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Time check - if close to timeout, return a deterministic fallback
    if not _elapsed_ok():
        improvement = f_best - mu
        z = improvement / sigma
        ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
        ei = np.maximum(ei, 0.0)
        return ei / cost

    # Thompson Sampling: Sample from the posterior distribution
    # This introduces stochasticity which helps escape local optima and explore high-uncertainty regions
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Calculate instantaneous improvement relative to f_best
    # If sample is better (lower) than f_best, improvement is positive
    improvement = f_best - samples
    
    # We only care about improvements (positive values). 
    # If the sample is worse than f_best, the score should be low (or zero).
    # Using max(0, improvement) ensures we don't pick points that are likely worse than current best.
    valid_improvement = np.maximum(0.0, improvement)
    
    # Cost-awareness: Divide by cost to prefer cheaper evaluations
    # This approximates Expected Improvement per Cost
    scores = valid_improvement / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
