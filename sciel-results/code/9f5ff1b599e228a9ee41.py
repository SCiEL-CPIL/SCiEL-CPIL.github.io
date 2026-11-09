"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Probability of Improvement (PoI) with an exploration bonus.
    
    Parameters:
    mu: array-like, shape (2000,), mean of GP predictions in objective units.
    sigma: array-like, shape (2000,), std of GP predictions in objective units.
    f_best: float, best observed objective value.
    cost: array-like, shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: array-like, shape (2000,), acquisition scores (higher is better).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Cost-adjusted PoI
    # We want high PoI and low cost
    score_main = poi / cost
    
    # Exploration bonus: encourages sampling where uncertainty is high
    # The coefficient 0.1 balances the scale of PoI (0-1) with sigma (objective units)
    # Assuming objective units are O(1-10), this is a moderate exploration pressure.
    exploration_bonus = 0.1 * sigma
    
    # Total acquisition score
    scores = score_main + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
