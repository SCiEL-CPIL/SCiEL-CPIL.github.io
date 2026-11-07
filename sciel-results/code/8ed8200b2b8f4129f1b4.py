"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-Effective Expected Improvement (CEI) with high gamma.
    
    Combines the robust Expected Improvement (EI) metric with the aggressive
    cost penalty (gamma=3.0) that proved successful in the UCB variant.
    
    Score = EI(x) / cost(x)^gamma
    
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
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety margin check
    if time.time() - t0 > 250:
        return np.ones_like(mu) / np.maximum(cost, 1e-6)
    
    # Parameters
    # Use the high gamma from the successful UCB program
    gamma = 3.0
    
    # Safe sigma for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm for PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical errors might cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-12)
    cost_penalty = cost_safe ** gamma
    
    # Final score
    score = ei / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety check
    if time.time() - t0 > 280:
         pass
         
    return score
# EVOLVE-BLOCK-END
