"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - best observed function value
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of improvement
    p_improve = norm.cdf(z)
    
    # Uncertainty bonus: encourages exploration in high-uncertainty regions
    # This helps find the global minimum which may be in high-cost areas
    uncertainty_bonus = p_improve * sigma_safe
    
    # Combined acquisition: EI + uncertainty bonus, normalized by sublinear cost penalty
    # Using 1/sqrt(cost) instead of 1/cost to be less punitive of high-cost regions
    scores = (ei + uncertainty_bonus) / np.sqrt(cost_safe)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
