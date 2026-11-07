# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines UCB (exploration) and EI (exploitation) with adaptive blending,
    normalized by evaluation cost.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: scalar, the lowest observed value so far
    - cost: array of shape (2000,) with evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = evaluate next)
    """
    # Deadline guard
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check if we're running too long (safety margin)
    if time.time() - t_start > 280.0:
        # Return a simple UCB-based score as fallback
        kappa = 1.0
        score = (f_best - mu + kappa * sigma_safe) / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB with tuned kappa
    # For Hartmann 6d, moderate exploration is beneficial
    kappa = 1.2
    ucb = f_best - mu + kappa * sigma_safe
    
    # Adaptive blending: when uncertainty is high relative to potential improvement,
    # lean more toward UCB; when we're confident, lean toward EI
    # blend factor in [0, 1], higher means more UCB weight
    uncertainty_ratio = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    blend = np.clip(uncertainty_ratio, 0.0, 1.0)
    
    # Smooth the blend to avoid abrupt transitions
    blend = 0.5 * blend
    
    # Combine EI and UCB
    combined = (1.0 - blend) * ei + blend * ucb
    
    # Normalize by cost
    scores = combined / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
