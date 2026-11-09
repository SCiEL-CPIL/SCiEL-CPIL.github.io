"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Hypervolume Improvement (EHVI) proxy.
    
    Uses a principled EHVI-based score that captures the expected improvement
    over the current best, normalized by evaluation cost. This approach
    naturally balances exploration and exploitation while respecting the
    cost structure of the problem.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, return a simple fallback
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-scores for improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) - the core of EHVI for single-objective
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EHVI proxy: weight EI by the probability of actual improvement
    # and add a small exploration bonus for high-uncertainty regions
    # that are also likely to improve
    p_improve = norm.cdf(z)  # P(f(x) < f_best)
    
    # The EHVI-like score emphasizes improvements that are both likely and large
    # Add a small uncertainty-driven exploration term for regions where
    # the GP is uncertain but there's still a chance of improvement
    exploration_weight = 0.1
    exploration = sigma * p_improve * (1.0 - p_improve)  # max at p=0.5
    
    # Combine: primary signal is EI, secondary is exploration in uncertain
    # regions that could still beat f_best
    score = (ei + exploration_weight * exploration) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
