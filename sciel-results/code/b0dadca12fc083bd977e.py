"""Cost-aware acquisition for Bayesian optimization on 4 test functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: budget-aware Thompson Sampling with cost normalization.
    
    Parameters:
    - mu: (2000,) array of GP means
    - sigma: (2000,) array of GP standard deviations
    - f_best: scalar, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of scores (higher = more desirable)
    """
    # Check deadline - if we're running out of time, return a fast fallback
    if _time_remaining() < 10.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Fast fallback: uncertainty-based score
        return sigma_safe / cost_safe

    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw samples from the posterior predictive
    # For each candidate, sample from N(mu, sigma^2)
    # The score is based on the predicted improvement over the cost
    
    # Sample from the posterior
    # Use a single random sample for efficiency
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Predicted improvement: how much better than f_best
    # For minimization, improvement = f_best - sample (positive if sample < f_best)
    improvement = f_best - samples
    
    # Cost-aware score: improvement per unit cost
    # This is the key idea: we want to maximize the expected improvement
    # relative to the cost of evaluation
    score = improvement / cost
    
    # Add a small exploration bonus based on uncertainty
    # This helps when f_best is close to the minimum
    # Weight the exploration term to be modest
    exploration = sigma / cost
    
    # Combine: primarily Thompson sample improvement, with some exploration
    # The exploration term encourages visiting uncertain regions
    score = 0.85 * score + 0.15 * exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
