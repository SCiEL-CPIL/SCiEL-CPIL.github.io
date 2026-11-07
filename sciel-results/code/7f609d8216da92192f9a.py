"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Uses Cost-Adjusted Expected Improvement (CAEI) with a small exploration bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is at least a small positive number to avoid division by zero
    # The theoretical min cost is 1, but we use 1e-6 for safety
    cost_safe = np.maximum(cost, 1e-6)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clamp EI to non-negative to handle numerical errors
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI
    # We prefer points with high expected improvement and low cost
    caei = ei / cost_safe
    
    # Add a small exploration term based on uncertainty, also cost-adjusted
    # This helps escape local minima or overconfident regions
    # alpha is a small weight for exploration
    alpha = 0.05
    exploration_term = alpha * sigma_safe / cost_safe
    
    score = caei + exploration_term
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
