"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a small sigma-weighted exploration term,
    normalized by evaluation cost.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and log issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero/negative costs
    # Cost is defined as 1 + mean(x_scaled) where x_scaled in [0,1], so cost in [1, 2]
    # But we guard against any anomalies
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting uncertain regions
    # This helps in early stages when the GP is uncertain about the location of the minimum
    # especially with shifted minimizers.
    beta = 0.1
    exploration = beta * sigma
    
    # Combine EI and exploration, then normalize by cost
    # Higher EI and higher uncertainty (sigma) increase the score.
    # Higher cost decreases the score.
    score = (ei + exploration) / cost
    
    # Ensure finite output; replace any non-finite values with 0.0
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
