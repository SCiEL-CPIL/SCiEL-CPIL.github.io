import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    Samples from the GP posterior and selects the candidate with the best
    predicted value per unit cost. This is a structurally different approach
    from EI: instead of computing an expected improvement metric, we draw a
    sample from the posterior and use it directly as the score.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative and add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample from the posterior for each candidate
    # We draw a single random number and use it to generate correlated samples
    # via the standard normal
    rng = np.random.default_rng()
    z_sample = rng.standard_normal(len(mu))
    
    # Sampled function values: f ~ N(mu, sigma^2)
    sampled_f = mu + sigma_safe * z_sample
    
    # For minimization, lower sampled_f is better
    # We want to maximize the score, so we negate the sampled value
    # and divide by cost to make it cost-aware
    
    # The score is -sampled_f / cost (higher is better)
    # This naturally balances exploration (high sigma leads to varied samples)
    # and exploitation (low mu leads to high scores)
    score = -sampled_f / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
