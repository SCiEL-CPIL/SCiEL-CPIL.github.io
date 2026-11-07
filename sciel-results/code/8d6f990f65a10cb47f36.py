"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a penalty for high uncertainty and a normalization by cost.

    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std dev predictions
        f_best: float - best observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost to get EI per unit cost
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-12)
    
    # Base score: EI per unit cost
    score = ei / cost_safe
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima in the acquisition landscape
    # Scale the uncertainty bonus to be smaller than typical EI values
    uncertainty_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine: weighted combination of EI/cost and uncertainty/cost
    # The weight on EI is higher to prioritize exploitation when it's clear
    final_score = score + uncertainty_bonus
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
