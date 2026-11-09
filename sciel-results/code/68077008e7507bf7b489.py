"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 320.0  # safety margin before 330s
    return _DEADLINE

def _time_left():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - evaluation costs
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # For standard normal, this is Phi(z)
    poi = norm.cdf(z)
    
    # Add a small uncertainty bonus to encourage exploration
    # The bonus is proportional to sigma but scaled down
    uncertainty_bonus = 0.1 * sigma
    
    # Combined score before cost adjustment
    combined = poi + uncertainty_bonus
    
    # Cost-aware normalization: divide by cost to prefer cheap evaluations
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
