"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition that combines Expected Improvement (EI) with
    a UCB exploration bonus, normalized by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Safety: handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration bonus: encourages exploring uncertain regions
    # The beta parameter balances exploitation (EI) vs exploration (UCB)
    beta = 1.0
    ucb_bonus = beta * sigma
    
    # Combined acquisition: EI + exploration bonus, divided by cost
    scores = (ei + ucb_bonus) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
