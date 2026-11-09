"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline safety
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement and UCB,
    tuned for cost-aware Bayesian optimization.
    
    Args:
        mu: Mean of GP predictions, shape (N,)
        sigma: Standard deviation of GP predictions, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each candidate, shape (N,)
        
    Returns:
        scores: Array of acquisition scores, shape (N,), higher = evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB with adaptive beta
    # Use a beta that balances exploration and exploitation
    # Scale beta based on the magnitude of sigma to keep UCB meaningful
    beta = 2.0  # Exploration-exploitation tradeoff
    
    # UCB term: mu + beta * sigma (for minimization, we want high values to be good)
    # But since we're minimizing, high mu is bad, so UCB for minimization is:
    # We want to find low values, so we use: -mu + beta * sigma (lower is better for mu)
    # Actually, for minimization, the acquisition should reward low mu and high uncertainty
    # Standard UCB for minimization: score = -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost to account for evaluation cost
    # Blend EI and UCB
    # Weight EI more when we have good information, UCB more when uncertain
    # Simple approach: use a weighted combination
    # Normalize EI and UCB to comparable scales
    
    # Scale EI by typical improvement magnitude
    ei_scale = np.std(improvement) + 1e-9
    ucb_scale = np.std(ucb) + 1e-9
    
    ei_normalized = ei / ei_scale
    ucb_normalized = ucb / ucb_scale
    
    # Blend: 60% EI, 40% UCB
    blended = 0.6 * ei_normalized + 0.4 * ucb_normalized
    
    # Make cost-aware
    scores = blended / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
