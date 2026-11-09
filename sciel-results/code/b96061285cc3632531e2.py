"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to enforce deadline if needed, though the function is stateless.
# The task says "Each evaluation must finish within 300 s". 
# The acquisition function is called 2000 times per iteration? No, "scores 2000 new uniform candidates".
# So it's called once per iteration with arrays of size 2000.
# The overhead of scipy.stats.norm.cdf on 2000 elements is negligible.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    Parameters:
    mu: array of mean predictions (2000,)
    sigma: array of standard deviation predictions (2000,)
    f_best: scalar, best value observed so far
    cost: array of evaluation costs (2000,)
    
    Returns:
    scores: array of acquisition values (2000,), higher is better
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Small epsilon to avoid division by zero or log issues
    eps = 1e-9
    
    # Calculate Z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If sigma is very small, z can be huge. 
    # We use a safe sigma.
    safe_sigma = np.maximum(sigma, eps)
    
    z = (f_best - mu) / safe_sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is the CDF of the standard normal distribution at z
    pi_values = norm.cdf(z)
    
    # Cost normalization
    # The directive suggests normalizing by cost^0.5
    # cost is in [1, 2] roughly (1 + mean(x_scaled) where x_scaled in [0,1])
    safe_cost = np.maximum(cost, eps)
    cost_factor = np.power(safe_cost, 0.5)
    
    scores = pi_values / cost_factor
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
