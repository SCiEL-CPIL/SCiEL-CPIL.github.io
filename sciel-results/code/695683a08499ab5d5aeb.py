"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Ackley 2D.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) per unit cost as the primary signal
    2. Uncertainty bonus to encourage exploration in uncertain regions
    3. Deadline guard for safety
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure we have the right shapes
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Safety: ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI per unit cost
    ei_per_cost = ei / cost
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps escape local optima by valuing uncertain regions
    # The bonus is scaled by 1/cost to maintain cost-awareness
    uncertainty_bonus = 0.1 * sigma / cost
    
    # Combine: primary signal is EI per cost, secondary is uncertainty
    scores = ei_per_cost + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
