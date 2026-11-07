"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI)
    with exploration bonuses and cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed objective value
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Exploration bonus: reward high uncertainty
    # Normalize sigma to [0, 1] range for consistent scaling
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Cost-aware weighting: prefer cheaper evaluations
    # Normalize cost to [0, 1] and use inverse as weight
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost weight: higher weight for lower cost
    # Use a smooth inverse: 1 / (1 + alpha * cost_norm)
    alpha = 2.0
    cost_weight = 1.0 / (1.0 + alpha * cost_norm)
    
    # Combine: PoI with exploration bonus, weighted by cost
    # Exploration bonus strength
    expl_weight = 0.3
    
    # Base score: PoI
    base_score = poi
    
    # Add exploration bonus
    exploration_bonus = expl_weight * sigma_norm
    
    # Combine and apply cost weighting
    scores = (base_score + exploration_bonus) * cost_weight
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
