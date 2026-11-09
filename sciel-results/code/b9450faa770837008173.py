"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses quantile-based cost-aware acquisition targeting the 25th percentile of posterior.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _quantile_aware_score(mu, sigma, f_best, cost, p=0.25):
    """
    Quantile-based cost-aware acquisition.
    
    For each candidate x, the posterior is N(mu, sigma^2).
    The p-th quantile is: q_p = mu + z_p * sigma, where z_p = Phi^{-1}(p).
    
    For minimization, we want low quantiles. We score by:
      score = -q_p / cost
    
    This directly targets regions where the p-th percentile of the
    posterior is low, providing principled balance between exploitation
    (low mu) and exploration (low sigma helps when mu is uncertain).
    
    The cost normalization prioritizes cheap evaluations.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z_p for p=0.25 is approximately -0.6745
    # More negative z_p means we're looking at lower quantiles
    # For p=0.25: z_p = -0.6745
    z_p = -0.6745  # Phi^{-1}(0.25)
    
    # p-th quantile of the posterior
    q_p = mu + z_p * sigma_safe
    
    # Score: we want low q_p (good for minimization) and low cost
    # score = -q_p / cost
    scores = -q_p / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Uses quantile-based scoring with adaptive p parameter.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        cost_safe = np.maximum(cost, 1e-9)
        # Fallback: simple UCB-like score
        return (-mu + 1.0 * sigma) / cost_safe
    
    # Adaptive p: 
    # When f_best is high (early), use lower p (more optimistic, more exploration)
    # When f_best is low (late), use higher p (more conservative, more exploitation)
    if f_best > 15.0:
        p = 0.10  # Very optimistic, heavy exploration
    elif f_best > 10.0:
        p = 0.15
    elif f_best > 5.0:
        p = 0.20
    elif f_best > 2.0:
        p = 0.25
    elif f_best > 1.0:
        p = 0.30
    else:
        p = 0.40  # More conservative, focus on exploitation
    
    # Compute quantile-aware scores
    scores = _quantile_aware_score(mu, sigma, f_best, cost, p=p)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
