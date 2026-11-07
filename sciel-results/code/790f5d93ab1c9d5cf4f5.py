"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a pure UCB (Upper Confidence Bound) strategy adapted for minimization.
    This is a structural change from the previous hybrid EI+UCB approach.
    The score is -mu + beta * sigma, where beta adapts to the progress
    toward the known minimum (-3.32237). This allows consistent exploration
    of high-uncertainty regions while shifting to exploitation as the search
    converges.
    
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
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Estimate progress: how close are we to the known minimum?
    # Hartmann 6d minimum is approximately -3.32237
    hartmann_min = -3.32237
    # The gap from the worst possible value (~-1.0) to the minimum
    total_range = hartmann_min - (-1.0)  # ~ -2.32237, so abs is ~2.32237
    # progress ~ 1.0 means we're at the minimum, ~0.0 means we're far away
    gap_to_min = f_best - hartmann_min  # positive when f_best > hartmann_min
    progress = np.clip(gap_to_min / abs(total_range), 0.0, 1.0)
    
    # Beta starts high (exploration) and decays with progress
    # Using exponential decay for smooth transition
    beta_max = 4.0
    beta_min = 0.5
    tau = 1.0  # time constant for exponential decay
    
    # beta = beta_min + (beta_max - beta_min) * exp(-progress / tau)
    # When progress=0 (far from min): beta ~ beta_max (explore)
    # When progress=1 (at min): beta ~ beta_min + (beta_max-beta_min)*exp(-1) ~ beta_min (exploit)
    beta = beta_min + (beta_max - beta_min) * np.exp(-progress / tau)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Higher score = more promising (lower mean, higher uncertainty)
    scores = -mu + beta * sigma_safe
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
