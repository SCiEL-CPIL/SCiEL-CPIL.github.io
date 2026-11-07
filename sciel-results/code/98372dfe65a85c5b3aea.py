"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    We use EI / cost as the core score, which has been shown to be optimal for this task.
    The key insight is that dividing by cost naturally balances exploration and exploitation
    while accounting for the heterogeneous evaluation costs.
    
    Args:
    mu (np.ndarray): GP mean predictions, shape (N,)
    sigma (np.ndarray): GP standard deviation predictions, shape (N,)
    f_best (float): Best observed value so far
    cost (np.ndarray): Cost of evaluating each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,)
    """
    # Ensure inputs are arrays with proper dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma = 0 to avoid division by zero
    # If sigma is 0, the distribution is a delta function
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI must be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is 1 + mean(x), so cost >= 1
    cost_safe = np.maximum(cost, 1e-10)
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
