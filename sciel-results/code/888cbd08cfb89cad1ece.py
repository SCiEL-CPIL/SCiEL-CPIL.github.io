"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a structural change: Thompson Sampling style with cost-aware scaling.
    Instead of EI + exploration, we sample from the posterior and select
    the point with the best expected value per unit cost.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float
    cost: array of shape (2000,)
    
    Returns:
    score: array of shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Structural change: Use a "virtual" sample that balances exploitation and exploration
    # based on the current best value and uncertainty.
    
    # Calculate a "confidence" term that encourages exploring high-uncertainty regions
    # but also exploits low-mean regions.
    
    # Normalize mu to have zero mean and unit variance for better scaling
    mu_mean = np.mean(mu)
    mu_std = np.std(mu)
    if mu_std < 1e-10:
        mu_std = 1.0
    
    mu_norm = (mu - mu_mean) / mu_std
    
    # Use a combination of normalized mean and sigma to create a score
    # that encourages both exploitation (low mu) and exploration (high sigma)
    
    # Weighting: alpha for exploitation, beta for exploration
    alpha = 1.0
    beta = 0.8
    
    # Score = -alpha * mu_norm + beta * sigma_safe
    # This creates a score where lower mu (better) and higher sigma (more uncertain) are favored
    
    raw_score = -alpha * mu_norm + beta * sigma_safe
    
    # Cost awareness: divide by cost to prefer cheaper evaluations
    # Use a softer penalty with gamma = 0.7
    gamma_cost = 0.7
    cost_safe = np.maximum(cost, 1e-8)
    
    score = raw_score / (cost_safe ** gamma_cost)
    
    # Sanitize
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
