# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
# The cost function is 1 + mean(x_scaled) where x_scaled = (x - (-4)) / 9
# We don't have x, but we have 'cost' which is 1 + mean(x_scaled)
# So mean(x_scaled) = cost - 1
# This doesn't directly help for the acquisition function unless we want to weight by cost.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) with predicted means
    - sigma: array of shape (2000,) with predicted standard deviations
    - f_best: float, the lowest observed value
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    # We want to maximize EI / cost
    # But we can also add a small exploration bonus to avoid always picking the cheapest point
    # A common approach is to use EI / (cost ** alpha) for some alpha in [0.5, 1.0]
    # Let's use EI / cost as in the seed, but with a small epsilon to avoid numerical issues
    
    # Alternative: Use a combination of EI and a penalty for high cost
    # scores = EI - beta * cost
    # But the seed uses EI / cost, which is a good balance
    
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
