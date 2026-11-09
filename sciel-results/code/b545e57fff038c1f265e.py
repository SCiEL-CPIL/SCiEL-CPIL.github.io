"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a UCB-style exploration bonus that scales with GP uncertainty
    and inversely with cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        GP mean predictions
    sigma : ndarray, shape (N,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (N,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores (higher = better)
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_per_cost = ei / cost
    
    # UCB exploration bonus: reward high uncertainty relative to cost
    # The bonus encourages exploring regions with high sigma
    # Scale the exploration term to be comparable to EI terms
    # Use a moderate coefficient; the ratio sigma/cost naturally balances
    # exploration in cheap, uncertain regions
    ucb_bonus = 0.5 * sigma_safe / cost
    
    # Combine: primary signal is cost-normalized EI, with exploration bonus
    scores = ei_per_cost + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
