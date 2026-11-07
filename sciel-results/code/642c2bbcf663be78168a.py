"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_EPS = 1e-9
_START_TIME = None
_DEADLINE = 250.0  # safety margin before 330s kill


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
    Acquisition function using a Bayesian Optimization under Budget (BOB) approach.
    Maximizes EI per unit cost, with a blended PI component for robustness.
    
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
    
    # Compute Expected Improvement
    ei = _compute_ei(mu, sigma, f_best)
    
    # Compute Probability of Improvement
    pi = _compute_pi(mu, sigma, f_best)
    
    # Cost-aware PI with exponential decay: favors low-cost points with high PI
    # The exponent beta controls how strongly cost penalizes high-cost candidates
    beta = 1.5
    pi_cost_aware = pi * np.exp(-beta * (cost - 1.0))
    
    # Cost-aware EI: EI divided by cost (marginal improvement per budget unit)
    ei_per_cost = ei / cost
    
    # Normalize both components to [0, 1]
    pi_ca_max = np.max(pi_cost_aware)
    if pi_ca_max < _EPS:
        pi_ca_max = 1.0
    pi_ca_norm = pi_cost_aware / pi_ca_max
    
    ei_pc_max = np.max(ei_per_cost)
    if ei_pc_max < _EPS:
        ei_pc_max = 1.0
    ei_pc_norm = ei_per_cost / ei_pc_max
    
    # Blend: 50% cost-aware PI (robust, smooth signal), 50% EI/cost (marginal return)
    # The exponential cost penalty on PI provides a smooth surrogate that
    # smoothly down-weights expensive candidates without hard thresholds
    scores = 0.5 * pi_ca_norm + 0.5 * ei_pc_norm
    
    # Check deadline
    if not _ensure_deadline():
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
