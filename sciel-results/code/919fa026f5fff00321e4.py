"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative Cost-Aware EI.
    
    Combines the probabilistic Expected Improvement (EI) with a multiplicative
    exploration factor. The score is PI * (1 + beta * sigma) / cost, where PI
    is the standard EI. This creates stronger coupling between exploitation and
    exploration than additive forms: exploration only amplifies regions that
    have a non-zero probability of improvement, while strongly boosting
    uncertain regions that do have improvement potential.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - _MARGIN:
        # Fallback: return something simple and fast
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe

    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Multiplicative exploration factor: (1 + beta * sigma)
    # This amplifies regions with high uncertainty, but only if they have
    # non-zero EI (since EI is multiplied, not added)
    beta = 1.0
    exploration_factor = 1.0 + beta * sigma_safe
    
    # Multiplicative cost-aware score
    scores = ei * exploration_factor / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
