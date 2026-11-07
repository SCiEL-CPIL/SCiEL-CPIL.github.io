# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Effective Expected Improvement (EI / cost).
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective EI: Divide by cost to prefer cheaper points
    # This directly optimizes improvement per unit cost
    scores = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps balance exploitation and exploration
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
