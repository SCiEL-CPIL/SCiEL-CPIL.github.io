"""Cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function with
    uncertainty penalty to prevent high-noise region dominance.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = evaluate next).
    """
    n = len(mu)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Expected Improvement
    # Improvement: f_best - mu (positive when mu < f_best)
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Clamp EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty penalty: penalize high sigma to avoid noisy regions
    # Use a scaled penalty that's significant in early sparse data
    # The penalty is proportional to sigma, which discourages points
    # with high uncertainty
    uncertainty_penalty = 0.3 * sigma_safe
    
    # Net score before cost normalization
    net_score = ei - uncertainty_penalty
    
    # Cost-aware: divide by cost to prefer cheaper points
    cost_safe = np.maximum(cost, 1e-12)
    
    scores = net_score / cost_safe
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
