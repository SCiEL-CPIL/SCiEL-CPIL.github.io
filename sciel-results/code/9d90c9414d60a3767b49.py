"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining expected improvement (EI)
    with a bonus for low-cost regions and an exploration term.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # But also add a small exploration bonus for uncertain regions
    # The cost is in [1, 2] since x is scaled to [0,1] and cost = 1 + mean(x_scaled)
    
    # Clip costs to avoid division by zero
    cost = np.maximum(cost, 1.0)
    
    # Cost-normalized EI
    ei_cost = ei / cost
    
    # Add a small exploration term: prefer regions with high uncertainty
    # relative to the range of f_best
    # Use a small weight to not dominate the EI signal
    f_range = max(abs(f_best), 1.0)
    exploration = sigma / f_range
    
    # Combine: primarily EI per cost, with a small exploration bonus
    # Weight exploration lightly
    scores = ei_cost + 0.1 * exploration
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
