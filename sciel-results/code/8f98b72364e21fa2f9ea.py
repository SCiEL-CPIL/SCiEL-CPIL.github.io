"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for GP computations to avoid redundant work across iterations
# We'll use a simple approach: since the function is called fresh each iteration
# with the same mu, sigma, cost arrays (2000 candidates), we just compute directly.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions for 2000 candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidate points
    f_best : float
        Best (lowest) observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next)
    """
    # Ensure we handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Add a small exploration term to balance exploitation and exploration
    # The EI/cost ratio can be very small when cost is high, so we add
    # a small bonus for high uncertainty to encourage exploration
    
    # Use a weighted combination: primarily EI/cost, but with a small
    # uncertainty bonus to avoid getting stuck in low-cost regions
    exploration_weight = 0.1
    uncertainty_bonus = exploration_weight * sigma / (sigma + 0.1)
    
    scores = ei / cost + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
