"""Cost-aware acquisition for Hartmann 6D with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 25.0  # seconds of safety margin before 330s kill

def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330.0 - _DEADLINE_MARGIN
    return _DEADLINE

def _time_exceeded():
    return time.time() >= _ensure_deadline()

def _expected_hypervolume_contribution(mu_i, sigma_i, f_ref, cost_i, cost_max=2.0):
    """
    Approximate the expected hypervolume contribution of a single candidate.
    
    We model the posterior as N(mu_i, sigma_i^2) for the objective f(x).
    The hypervolume contribution of a point (f, c) relative to reference point
    (f_ref, cost_max) is max(0, (f_ref - f)) * max(0, (cost_max - c)).
    
    E[HV_contrib] = E[max(0, f_ref - F) * max(0, cost_max - C)]
    Since C is deterministic given x, this is:
    E[max(0, f_ref - F)] * max(0, cost_max - C)
    = (f_ref - mu_i) * Phi(z) + sigma_i * phi(z)  where z = (f_ref - mu_i)/sigma_i
    multiplied by max(0, cost_max - C).
    
    But we also want to reward points that are cheap AND good. The EHV approach
    naturally does this because the cost_max - C factor penalizes expensive points.
    """
    sigma_safe = max(sigma_i, 1e-10)
    z = (f_ref - mu_i) / sigma_safe
    # Expected improvement part: E[max(0, f_ref - F)]
    ei = (f_ref - mu_i) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost factor: how much "room" does this cost leave in the budget
    # Points with lower cost get higher weight
    cost_factor = max(0.0, cost_max - cost_i)
    
    return ei * cost_factor

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Hypervolume (EHV) approximation.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviation predictions
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns: (2000,) array of finite scores (higher = evaluate next)
    """
    _ensure_deadline()
    
    # Clamp sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Reference point for hypervolume: use f_best as the objective reference
    # and 2.0 as the max possible cost (cost = 1 + mean(x) in [0,1]^6, so cost in [1,2])
    f_ref = f_best
    cost_max = 2.0
    
    # Compute EHV contribution for each candidate
    # E[max(0, f_ref - F)] where F ~ N(mu, sigma^2)
    z = (f_ref - mu) / sigma
    ei = (f_ref - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost factor: rewards cheaper evaluations
    # cost is in [1, 2], so (cost_max - cost) is in [0, 1]
    cost_factor = np.maximum(0.0, cost_max - cost)
    
    # EHV score = EI * cost_factor
    # This naturally balances: high EI (good expected improvement) 
    # AND low cost (more budget remaining)
    scores = ei * cost_factor
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the acquisition landscape
    # Weight by uncertainty relative to the scale of f_best
    uncertainty_scale = max(abs(f_best), 1e-6)
    exploration_bonus = 0.1 * sigma / uncertainty_scale
    
    scores = scores + exploration_bonus * cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
