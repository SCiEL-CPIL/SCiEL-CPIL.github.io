"""Rewritten program using Quantile of the Posterior (QoP) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = None
_TIME_LIMIT = 320.0  # Safety margin before 330s kill switch

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Quantile of the Posterior (QoP).
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (2000,) with evaluation costs.
        
    Returns:
        Array of shape (2000,) with scores (higher is better).
    """
    global _start_time
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: if we are close to the limit, return a fast fallback
    # This prevents timeouts in the evaluator loop
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: simple cost-aware mean minimization
        # Ensure finite output
        fallback_scores = -mu / cost
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # QoP Parameter: 0.25 (25th percentile)
    # This targets the lower tail of the posterior, balancing exploitation (low mu)
    # and exploration (high sigma reduces the quantile for p < 0.5)
    p = 0.25
    
    # Precompute the z-score for the quantile
    # norm.ppf(0.25) is approximately -0.67449
    z_p = norm.ppf(p)
    
    # Calculate the Quantile of the Posterior for each candidate
    # Q_p(x) = mu(x) + sigma(x) * z_p
    # Since z_p is negative, higher sigma leads to lower Q_p (more attractive)
    q_p = mu + sigma * z_p
    
    # We want to minimize Q_p, so we maximize -Q_p
    # Normalize by cost to prioritize cheaper evaluations
    # Cost is in [1, 2], so division is safe
    scores = -q_p / cost
    
    # Ensure all outputs are finite to satisfy validity constraints
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
