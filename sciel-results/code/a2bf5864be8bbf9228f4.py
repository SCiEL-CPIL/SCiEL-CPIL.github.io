"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Improvement (CEI) acquisition function.
    
    Computes the Expected Improvement (EI) for each candidate point and
    normalizes it by the evaluation cost. This encourages the algorithm
    to find points that are likely to improve the best observed value
    while penalizing expensive evaluations.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return 1.0 / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is CDF, phi is PDF of standard normal distribution
    
    z = (f_best - mu) / sigma_safe
    
    # Handle numerical issues in z
    z = np.where(np.isfinite(z), z, 0.0)
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
