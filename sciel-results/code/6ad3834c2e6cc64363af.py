"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PII).
    
    This approach focuses on maximizing the probability of finding a new best
    value per unit cost, rather than the expected magnitude of improvement.
    This is structurally different from EI/UCB blends and can be more robust
    in tight budget settings by prioritizing high-confidence gains.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Compute Probability of Improvement (PII)
    # PII = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    pii = norm.cdf(z)
    pii = np.where(np.isfinite(pii), pii, 0.0)
    
    # Cost-aware scoring: PII per unit cost
    score = pii / cost
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Add a small exploration component based on uncertainty
    # We use a fixed small weight to avoid distorting the PII signal too much
    # but ensure we don't completely ignore high-uncertainty regions if they
    # have reasonable PII.
    # A simple exploration term: sigma normalized by max sigma
    max_sigma = np.max(sigma)
    if max_sigma > 1e-9:
        exploration = sigma / max_sigma
    else:
        exploration = np.zeros_like(sigma)
    
    # Blend PII/cost with exploration
    # Early on, we might want more exploration, but PII already captures
    # some of that via the tails. We keep the weight low.
    w_explore = 0.1
    
    scores = (1.0 - w_explore) * score + w_explore * (exploration / (cost + 1e-9))
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
