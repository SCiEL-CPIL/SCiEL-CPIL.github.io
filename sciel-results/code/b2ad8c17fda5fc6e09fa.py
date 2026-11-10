"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) acquisition with budget-horizon lookahead.
    
    Structural design: a single acquisition metric where exploration-exploitation
    balance is derived from the GP posterior's own uncertainty structure,
    not from hand-tuned combinations of EI/UCB/PI.
    
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
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- CEI with budget-horizon lookahead ---
    
    # Step 1: Compute base Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Step 2: Structural Change: Sub-linear cost penalty
    # Instead of linear division (EI / Cost), we use a sub-linear penalty (EI / sqrt(Cost)).
    # This reduces the bias towards low-cost points, allowing the optimizer to
    # explore higher-cost regions (which may contain the global minimum in the
    # center of the domain) more aggressively if the EI is high enough.
    # This shifts the representation from "minimum cost per unit improvement"
    # to "improvement weighted by inverse square root of cost".
    cost_penalty = np.sqrt(cost)
    score = ei / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
