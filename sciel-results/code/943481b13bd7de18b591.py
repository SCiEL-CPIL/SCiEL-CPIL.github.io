import numpy as np
from scipy.stats import norm
import time

# Time guard: track when we started to avoid exceeding the 330s wall-clock limit
_ACQ_START = None
_TIME_LIMIT = 325.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses Expected Improvement divided by cost (EI/cost), which is the
    standard cost-aware acquisition for budgeted BO.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    global _ACQ_START
    
    # Deadline guard
    if _ACQ_START is None:
        _ACQ_START = time.time()
    elapsed = time.time() - _ACQ_START
    if elapsed > _TIME_LIMIT:
        # Return a simple fallback to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.maximum(cost, 1e-10)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI / cost
    # This is the standard cost-aware EI (cEI)
    cost_safe = np.maximum(cost, 1e-10)
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
