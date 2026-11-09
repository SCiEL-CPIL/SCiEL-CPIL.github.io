"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) signals, normalized by evaluation cost.
    
    This approach balances exploitation (finding points likely to improve
    on the current best) with exploration (visiting uncertain regions)
    while respecting the non-uniform cost structure of the problem.
    
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
        # Fallback: Pure EI / Cost
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI) with a soft floor to prevent collapse
    # EI is the expected improvement over f_best.
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Add a small exploration bonus proportional to sigma to prevent EI from
    # collapsing to zero when mu is close to f_best in high-uncertainty regions
    ei_floored = ei + 0.1 * sigma * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) with adaptive beta
    # Beta increases as f_best approaches the known minimum (0 for most functions),
    # encouraging more exploration near the optimum
    # Scale: when f_best is small (close to 0), we want more exploration
    f_best_abs = abs(f_best)
    # Adaptive beta: starts at 0.3 when f_best is large, increases to 1.0 when f_best ~ 0
    beta = 0.3 + 0.7 * np.exp(-f_best_abs)
    ucb = -mu + beta * sigma
    
    # 3. Combine signals
    # EI is the primary signal (directly tied to improvement), UCB provides exploration
    score_raw = ei_floored + 0.4 * ucb
    
    # 4. Cost-aware normalization using log-cost for smoother penalty
    # Using log(1+cost) instead of raw cost gives a smoother, less extreme penalty
    cost_norm = np.log1p(cost)
    score = score_raw / cost_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
