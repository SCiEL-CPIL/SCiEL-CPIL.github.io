"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with
    a variance-aware exploration term, normalized by evaluation cost.
    
    The acquisition score is:
      (EI + beta * sigma) / cost
    
    where EI is the expected improvement over f_best and beta controls
    the exploration-exploitation tradeoff. The cost normalization penalizes
    expensive evaluations.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # For minimization, improvement is f_best - Y, so we want Y < f_best
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we need to be careful: when mu > f_best, z < 0, and EI should be small
    # Standard EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # This is correct: when mu << f_best, z >> 0, Phi(z) ~ 1, phi(z) ~ 0, EI ~ f_best - mu > 0
    # When mu >> f_best, z << 0, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration term to encourage visiting unexplored regions
    # The sigma term already provides some exploration, but we can add a small
    # bonus for high uncertainty
    beta = 0.1
    exploration = beta * sigma_safe
    
    # Combined acquisition before cost normalization
    acq = ei + exploration
    
    # Ensure non-negative
    acq = np.maximum(acq, 0.0)
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = acq / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
