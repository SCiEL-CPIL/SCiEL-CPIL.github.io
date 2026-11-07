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
    Acquisition function using Expected Improvement per unit cost (BOB principle)
    blended with cost-aware Probability of Improvement.
    
    The EI/cost formulation directly optimizes marginal return per unit of budget,
    which is the principled Bayesian Optimization under Budget approach. This
    naturally balances high-improvement exploitation with budget efficiency.
    The PI component with exponential cost penalty provides a secondary signal
    for high-confidence cheap improvements.
    
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
    
    # Primary signal: EI / cost (BOB principle)
    # This directly maximizes expected improvement per unit of budget spent
    ei_per_cost = ei / cost
    
    # Secondary signal: PI with exponential cost penalty
    # Sharp preference for cheap, high-probability improvement points
    beta = 2.0
    cost_penalty = np.exp(-beta * cost)
    pi_cost_aware = pi * cost_penalty
    
    # Normalize both components to [0, 1] to prevent scale dominance
    ei_max = np.max(ei_per_cost)
    if ei_max < _EPS:
        ei_max = 1.0
    ei_norm = ei_per_cost / ei_max
    
    pi_max = np.max(pi_cost_aware)
    if pi_max < _EPS:
        pi_max = 1.0
    pi_norm = pi_cost_aware / pi_max
    
    # Blend: 70% EI/cost (primary BOB signal), 30% cost-aware PI (secondary)
    # EI/cost is the principled budget-aware acquisition
    # PI with exponential penalty adds exploitation of cheap, confident improvements
    scores = 0.7 * ei_norm + 0.3 * pi_norm
    
    # Check deadline
    if not _ensure_deadline():
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
