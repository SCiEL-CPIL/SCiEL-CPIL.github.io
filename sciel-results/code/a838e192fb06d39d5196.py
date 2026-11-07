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
    Cost-aware acquisition function using Cost-Aware Expected Improvement (EI).
    
    For minimization, Expected Improvement is defined as:
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) 
           + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    We then divide by the cost to get the cost-aware score.
    
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
        # Prefer low cost and high uncertainty
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If sigma is very small, z can be large. We clip it to avoid numerical issues.
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10) # Clip to prevent overflow in exp
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Phi(z) is the CDF, phi(z) is the PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but numerical errors can occur)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
