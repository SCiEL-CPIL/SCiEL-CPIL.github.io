# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Approach: Cost-Weighted Probability of Improvement (PI) for minimization.
    # PI quantifies the probability that a new evaluation will improve upon f_best.
    # Unlike EI, it does not scale by the expected magnitude of improvement,
    # focusing instead on the likelihood of success.
    # We divide by cost to prefer cheaper evaluations that offer similar probabilities of improvement.
    
    # Ensure sigma is positive to avoid division by zero in Z calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Z score: (f_best - mu) / sigma
    # If f_best > mu, Z is positive (we expect improvement)
    # If f_best < mu, Z is negative (we expect no improvement)
    Z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement formula for minimization:
    # PI = Phi(Z)
    # Where Phi is the CDF of the standard normal distribution.
    
    # Use scipy.stats.norm for numerical stability
    PI = norm.cdf(Z)
    
    # Clip to [0, 1] to handle floating point errors
    PI = np.clip(PI, 0.0, 1.0)
    
    # Normalize by cost to get probability per unit cost
    score = PI / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
