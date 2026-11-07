"""Rewritten program using multiplicative Probability of Improvement with soft cost penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (PI)
    with a soft multiplicative cost penalty.
    
    PI = Phi((f_best - mu) / sigma)
    Score = PI^cost
    
    This naturally saturates at 1.0 for certain improvements and decays with uncertainty.
    The exponent cost penalizes expensive evaluations multiplicatively, favoring
    cheap, high-confidence improvements.
    """
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores for probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Clip to avoid log(0) issues; PI is in [0, 1]
    pi = np.clip(pi, 1e-15, 1.0)
    
    # Soft cost penalty via exponentiation: PI^(1/cost)
    # This reduces the penalty for expensive points compared to PI^cost,
    # allowing the search to explore higher-cost regions if the PI is high enough,
    # which is crucial if the true minimum is not in the cheapest region.
    cost_safe = np.maximum(cost, 1e-9)
    exponent = 1.0 / cost_safe
    score = np.power(pi, exponent)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
