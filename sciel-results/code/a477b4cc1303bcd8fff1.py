# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses Lower Confidence Bound (LCB) for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    t_start = time.time()
    deadline = t_start + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-10)

    # LCB for minimization: we want to minimize, so lower confidence bound is
    # lcb = mu - beta * sigma
    # We want to select points with LOW lcb (more negative = more promising)
    # So score = -lcb = -mu + beta * sigma (higher is better)
    
    # Adaptive beta: larger exploration early, smaller late
    # f_best starts high (~100+ for Beale), minimum is 0
    if f_best > 50:
        beta = 2.5
    elif f_best > 10:
        beta = 2.0
    elif f_best > 3:
        beta = 1.5
    elif f_best > 1:
        beta = 1.0
    else:
        beta = 0.5

    # LCB score: higher is better
    scores = -mu + beta * sigma_safe

    # Add a small EI component to help with exploitation
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -12, 12)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(improvement > 0, ei, 0.0)
    
    # Scale EI to match the scale of LCB scores
    ei_scale = np.std(scores) + 1e-10
    scores += 0.3 * (ei / ei_scale) * np.std(scores)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
