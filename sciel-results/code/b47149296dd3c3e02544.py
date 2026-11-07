"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    divided by cost.
    
    Args:
        mu: Mean predictions, shape (N,)
        sigma: Standard deviation predictions, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        Scores for each candidate, shape (N,)
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # EI >= 0, higher is better
    
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats.norm for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize by cost to favor cheaper evaluations
    scores = ei / cost_safe
    
    # Replace non-finite values with 0 (no improvement expected)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
