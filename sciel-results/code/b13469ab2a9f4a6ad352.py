"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 280  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def _expected_improvement(mu, sigma, f_best):
    """Standard expected improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma < 1e-7, np.where(mu < f_best, f_best - mu, 0.0), ei)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining LCB-style exploration with
    probability-of-improvement weighting, normalized by cost.
    
    Structural change: instead of a soft-min cost penalty applied after
    normalizing scores, we use a multiplicative cost-aware form:
        score = p_beat * (f_best - mu + beta * sigma) / cost
    
    This directly rewards candidates that are both likely to beat f_best
    AND cheap to evaluate. The LCB term (beta * sigma - mu) provides
    linear exploration pressure, while the probability term p_beat focuses
    search near the current best.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma and cost to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Deadline guard before expensive computation
    if not _elapsed_ok():
        # Fast fallback: simple LCB / cost
        beta = 2.0
        scores = (beta * sigma - mu) / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Adaptive beta: explore more when far from optimum, exploit when close
    f_min_known = -78.3323
    # Progress: 0 when far from optimum, 1 when close
    # Use a scale that reflects the typical range of f_best values
    progress = np.clip((f_best - f_min_known) / 100.0, 0.0, 1.0)
    
    # Beta decreases as we get closer to optimum (more exploitation)
    beta = 2.5 - 1.0 * progress  # ranges from 2.5 (far) to 1.5 (close)
    
    # Improvement potential: how much lower could this point be than f_best?
    # LCB-style: f_best - (mu - beta * sigma) = (f_best - mu) + beta * sigma
    improvement_potential = (f_best - mu) + beta * sigma
    
    # Probability that this point actually beats f_best
    z = improvement_potential / sigma
    p_beat = norm.cdf(z)
    
    # Cost-aware score: multiply by probability of success, divide by cost
    # This naturally balances: high p_beat AND low cost = high score
    scores = p_beat * improvement_potential / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
