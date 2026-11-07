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
    Cost-aware acquisition function using a log-barrier POI with sublinear uncertainty
    and softened cost penalty.
    
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
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for probability of improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Log-barrier on PI: log1p(PI) saturates exploitation smoothly
    # This prevents over-concentration on near-certain improvements
    log_pi = np.log1p(pi)
    
    # Sublinear exploration term: sigma^0.7 provides balanced exploration
    # across uncertainty scales without being dominated by extreme sigma
    sigma_explore = np.power(sigma_safe, 0.7)
    
    # Softened cost penalty: cost^0.5 allows visiting slightly more expensive
    # but much more informative regions
    cost_penalty = np.power(cost_safe, 0.5)
    
    # Combined score: log-barrier PI * sublinear uncertainty / softened cost
    scores = log_pi * sigma_explore / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
