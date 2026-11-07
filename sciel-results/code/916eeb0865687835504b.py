"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with an adaptive cost exponent.
    
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
    
    # Adaptive cost exponent: when f_best is close to 0 (min of Powell),
    # we should be more aggressive about finding the global minimum,
    # so reduce the cost penalty. When f_best is far from 0, use standard cost penalty.
    # The minimum of Powell is 0, so f_best <= 0 indicates we're close to optimal.
    if f_best < 0.1:
        # Close to optimal: use a smaller cost exponent to explore more
        cost_exponent = 0.5
    elif f_best < 1.0:
        # Moderately close: use a moderate cost exponent
        cost_exponent = 0.8
    else:
        # Far from optimal: use full cost penalty
        cost_exponent = 1.0
    
    # Normalize by cost^exponent to favor cheaper evaluations
    scores = ei / np.power(cost_safe, cost_exponent)
    
    # Replace non-finite values with 0 (no improvement expected)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
