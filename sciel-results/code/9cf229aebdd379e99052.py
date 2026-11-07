"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute a lookup table for the normal CDF/PDF to speed up acquisition
# We'll use a vectorized approach with norm.cdf and norm.pdf which are fast enough
# for 2000 points, but let's ensure we handle edge cases.

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : np.ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = ei / cost_safe
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
