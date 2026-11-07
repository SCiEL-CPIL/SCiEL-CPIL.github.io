"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed value so far (lowest since we're minimizing).
    cost : array-like, shape (2000,)
        Cost of evaluating each point, shape (2000,).
        Cost = 1 + mean(x scaled to [0, 1]) where x is in [-10, 10]^2.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    # Deadline guard
    t0 = time.time()
    deadline = 295  # safety margin before 300s limit
    
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    cdf_vals = norm.cdf(z_clipped)
    pdf_vals = norm.pdf(z_clipped)
    
    # EI for minimization
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Cost-aware: EI per unit cost
    scores_ei = ei / cost_safe
    
    # Time check after EI computation
    if time.time() - t0 > deadline:
        scores = np.where(np.isfinite(scores_ei), scores_ei, 0.0)
        return scores
    
    # Bounded exploration bonus: sigma / (sigma + 0.1)
    # This is bounded in [0, 1) and encourages visiting uncertain regions
    # without dominating the EI signal
    exploration = sigma_safe / (sigma_safe + 0.1)
    
    # Cost-aware exploration: scale by 1/cost to respect the cost budget
    scores_explore = exploration / cost_safe
    
    # Combine EI and exploration with a small weight for exploration
    # The exploration term provides a smooth, bounded boost to uncertain areas
    exploration_weight = 0.1
    scores = scores_ei + exploration_weight * scores_explore
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
