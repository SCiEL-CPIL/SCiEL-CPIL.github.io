"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-Effective Expected Improvement (CEI).
    
    Score = EI(x) / cost(x)^gamma
    
    EI is computed for minimization.
    gamma=2.0 balances exploration and cost efficiency.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative (mu > f_best), EI should be 0.
    # The formula above naturally yields small positive or negative values, 
    # but strictly EI is max(0, ...).
    # However, for large negative z, Phi(z) is small, phi(z) is small.
    # Let's explicitly clamp to 0.
    
    ei_term = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative EI
    ei = np.maximum(ei_term, 0.0)
    
    # Cost penalty
    gamma = 2.0
    cost_penalty = cost_safe ** gamma
    
    # Final score
    score = ei / cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
    # EVOLVE-BLOCK-END
