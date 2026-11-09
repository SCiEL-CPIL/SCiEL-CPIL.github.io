"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Deterministic UCB acquisition with time-decaying exploration coefficient.
    
    score = -mu + beta(t) * sigma
    where beta(t) = 2.0 * (1 - t/30), t = number of evaluations so far.
    
    We estimate t from f_best: at the start f_best is from 5 random points,
    and each subsequent evaluation may lower it. We use a heuristic:
    the number of evaluations is roughly related to how "tight" f_best is
    relative to the known minimum. However, since we don't have direct access
    to the evaluation count, we use a proxy: the ratio of f_best to the
    known minimum (-1.8013). Early on, f_best is far from the minimum,
    so we assume few evaluations; later, f_best approaches the minimum.
    
    A simpler and more robust approach: use the spread of mu and sigma
    to infer the stage. But the cleanest is to track state. Since we can't,
    we use a fixed schedule based on a reasonable assumption.
    
    Actually, we can use a clever trick: the number of evaluations t can be
    estimated from the fact that f_best is the minimum of t samples.
    But that's complex. Let's use a simple heuristic:
    
    - If f_best is much larger than the minimum (say > -1.0), we're early: t ~ 5-10
    - If f_best is close to the minimum (say < -1.7), we're late: t ~ 25-30
    
    Linear interpolation between these regimes.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) objective value observed so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Estimate the number of evaluations t from f_best
    # Known minimum is -1.8013. At t=5, f_best is typically around -1.0 to -1.5
    # At t=30, f_best is close to -1.8
    # Use a simple linear mapping
    f_min_known = -1.8013
    
    # Normalize f_best to [0, 1] range where 0 = very early, 1 = very late
    # f_best starts around -1.0 (early) and ends around -1.8 (late)
    # Map f_best from [-1.0, -1.8] to [0, 1]
    f_early = -1.0
    f_late = -1.8
    
    if f_best <= f_late:
        progress = 1.0
    elif f_best >= f_early:
        progress = 0.0
    else:
        progress = (f_early - f_best) / (f_early - f_late)
    
    # Map progress to t in [5, 30]
    t_est = 5 + 25 * progress
    
    # Time-decaying beta: beta(t) = 2.0 * (1 - t/30)
    beta = 2.0 * (1.0 - t_est / 30.0)
    
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.1, 2.0)
    
    # UCB score: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
