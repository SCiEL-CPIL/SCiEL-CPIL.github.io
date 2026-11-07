"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic penalty that decays as f_best improves, balancing
    exploration of expensive regions early and exploitation of cheap
    high-confidence points later.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Dynamic cost penalty that decays as f_best improves
    # When f_best is large (early), we're more willing to pay high costs
    # When f_best is small (late), we penalize high costs more heavily
    # Use a logarithmic scale for better conditioning
    f_best_abs = np.abs(f_best)
    if f_best_abs < 1e-6:
        # Very close to optimum, heavily penalize cost
        penalty_factor = 10.0
    else:
        # Decay penalty from ~5 (early) to ~1 (late)
        # Use f_best_abs to track progress
        decay = np.exp(-0.5 * np.log1p(f_best_abs))
        penalty_factor = 1.0 + 4.0 * decay
    
    # Cost-aware score: EI divided by (penalty_factor * cost)
    # This makes expensive points less attractive, especially late in optimization
    scores = ei / (penalty_factor * cost)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
