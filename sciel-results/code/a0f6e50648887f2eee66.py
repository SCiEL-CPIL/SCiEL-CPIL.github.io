"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_EPS = 1e-9
_START_TIME = None
_DEADLINE = 280.0  # safety margin before 330s kill


def _ensure_deadline():
    """Return True if we should continue, False if we've hit the safety deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < _DEADLINE


def _compute_pi(mu, sigma, f_best):
    """Compute Probability of Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    return np.maximum(pi, 0.0)


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma * pdf
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function dominated by cost-aware Probability of Improvement (PI),
    with a small contribution from cost-aware Expected Improvement (EI/cost).
    
    The PI * exp(-beta * cost) term strongly favors points that are both cheap 
    and likely to improve upon the current best. This has shown to be superior 
    to EI-based acquisitions in this cost-aware setting. The EI/cost term is 
    included at a low weight to provide a secondary exploitation signal for 
    points with high expected gain per unit cost, but it does not dominate the 
    search.
    
    Args:
        mu: numpy array of shape (2000,), predicted means
        sigma: numpy array of shape (2000,), predicted standard deviations
        f_best: float, best observed value so far
        cost: numpy array of shape (2000,), evaluation costs for each candidate
        
    Returns:
        numpy array of shape (2000,), acquisition scores (higher is better)
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Convert to numpy arrays for consistent handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Compute Probability of Improvement
    pi = _compute_pi(mu, sigma, f_best)
    
    # Compute Expected Improvement
    ei = _compute_ei(mu, sigma, f_best)
    
    # Primary signal: PI with exponential cost penalty
    # beta=2.0 was found to be effective in the top-performing pure PI strategy
    beta = 2.0
    cost_penalty = np.exp(-beta * cost)
    pi_cost_aware = pi * cost_penalty
    
    # Secondary signal: EI / cost (BOB principle)
    ei_per_cost = ei / cost
    
    # Normalize both components to [0, 1] to allow blending
    pi_max = np.max(pi_cost_aware)
    if pi_max < _EPS:
        pi_max = 1.0
    pi_norm = pi_cost_aware / pi_max
    
    ei_max = np.max(ei_per_cost)
    if ei_max < _EPS:
        ei_max = 1.0
    ei_norm = ei_per_cost / ei_max
    
    # Blend: 85% cost-aware PI, 15% cost-aware EI
    # Heavily favor the PI*exp(-cost) signal which has proven more effective
    scores = 0.85 * pi_norm + 0.15 * ei_norm
    
    # Check deadline
    if not _ensure_deadline():
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
