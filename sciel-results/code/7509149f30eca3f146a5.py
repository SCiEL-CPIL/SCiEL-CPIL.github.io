"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Expected Improvement (EI) acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : np.ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute EI
    # Use numerical stability: clip z to avoid extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: for minimization, we want to find points where f(x) < f_best
    # The improvement is f_best - f(x), so EI = E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost: prefer candidates that give good EI per unit cost
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
