"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe default
    elapsed = _get_elapsed()
    if elapsed > 320:
        return np.zeros(len(mu), dtype=np.float64)
    
    n = len(mu)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # Compute the z-score for Expected Improvement
    z = (f_best - mu) / sigma_clipped
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Weighted Thompson Sampling approach:
    # Sample a single global multiplier lambda from a Beta distribution
    # to stochastically shift exploration-exploitation trade-off per iteration
    
    # Use a Beta distribution that favors moderate exploitation with some exploration
    # Beta(2, 3) has mean 0.4, encouraging balance between EI and uncertainty
    np.random.seed(42 + int(elapsed * 100))  # Deterministic but varies over time
    lambda_val = np.random.beta(2, 3)
    
    # Compute composite score: (lambda * EI + (1-lambda) * sigma) / cost
    # This stochastically shifts between exploitation (EI) and exploration (sigma)
    composite = lambda_val * ei + (1 - lambda_val) * sigma_clipped
    
    # Divide by cost to make it cost-aware
    # Cost is in [1, 2], so division normalizes for evaluation expense
    scores = composite / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
