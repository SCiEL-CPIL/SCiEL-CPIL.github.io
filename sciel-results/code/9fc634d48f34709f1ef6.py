"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a parametric model optimized against task objective.
    
    The approach uses Expected Improvement with an optimized exploration-exploitation
    balance. The exploration weight is tuned to work well for 6D Hartmann function
    with the given GP settings.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Start a timer to guard against the 330s wall-clock limit
    t_start = time.time()
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Structural Change: Switch from Expected Improvement (EI) to Lower Confidence Bound (LCB).
    # For minimization, we want to find points where the function value is likely to be low.
    # The LCB is mu - beta * sigma. We want to MINIMIZE the LCB.
    # However, the acquisition function must return scores where HIGHER is better.
    # Therefore, we maximize the negative LCB: score = -mu + beta * sigma.
    #
    # This is linear in mu and sigma, unlike EI which is non-linear.
    # It does not explicitly depend on f_best in the score calculation,
    # making it robust to the specific current best value and focusing on
    # the potential of the region defined by the GP posterior.
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages more exploration (higher sigma regions).
    # For 6D Hartmann, a moderate-to-high beta is often effective to escape local optima.
    beta = 1.5
    
    scores = -mu + beta * sigma_safe
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s
        # Return what we have; it should be finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
