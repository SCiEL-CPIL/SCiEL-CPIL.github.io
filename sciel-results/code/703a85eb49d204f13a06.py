"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending UCB and Expected Improvement (EI).
    
    Args:
        mu: Array of shape (N,) with predicted means.
        sigma: Array of shape (N,) with predicted standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    # UCB for minimization: -mu + beta * sigma
    # Use a moderate beta that balances exploration and exploitation
    beta = 1.0
    ucb_scores = -mu + beta * sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend UCB and EI with equal weight
    blended_scores = 0.5 * ucb_scores + 0.5 * ei_scores
    
    # Make cost-aware by dividing by cost
    cost_safe = cost + epsilon
    scores = blended_scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
