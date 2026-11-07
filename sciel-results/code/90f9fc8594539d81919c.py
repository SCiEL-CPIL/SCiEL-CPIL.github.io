"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed function value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Expected Improvement with small xi for exploration
    xi = 0.005
    improvement = f_best - mu - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide EI by cost
    # Use a softened cost to avoid extreme penalties
    cost_soft = cost + 0.1
    
    # Add exploration bonus based on uncertainty
    # This helps escape local minima in Rastrigin
    # Scale exploration relative to the scale of the function
    # Use a normalized uncertainty term
    sigma_norm = sigma / (sigma + 0.5)
    exploration = 0.1 * sigma_norm
    
    # Combine: cost-aware EI + exploration
    score = ei / cost_soft + exploration
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
