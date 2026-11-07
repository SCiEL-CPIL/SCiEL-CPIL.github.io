import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time at the start of each acquisition call and bail out early
# if we're getting too close to the limit, returning a safe fallback.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and an
    adaptive UCB exploration term for Bayesian optimization of
    Goldstein-Price 2D.
    
    The EI component provides strong exploitation toward the best observed
    region, while the UCB component (with adaptive beta) encourages
    exploration in uncertain areas. The blend is weighted to favor
    exploitation as the search matures.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    global _start_time, _DEADLINE_MARGIN
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as a fast fallback
        return np.nan_to_num(-mu + 2.0 * sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- Adaptive UCB component ---
    # Use the spread of mu as a proxy for global uncertainty.
    # Early on, mu varies a lot -> higher beta (more exploration).
    # Later, mu converges -> lower beta (more exploitation).
    mu_std = np.std(mu)
    # Map mu_std to beta in roughly [1.0, 2.5]
    beta = 1.0 + 1.5 * (mu_std / (mu_std + 0.3))
    
    # UCB score for minimization: we want to maximize -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # --- Normalize both components to [0, 1] for fair blending ---
    # Normalize EI
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-10:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize UCB score
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-10:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # --- Blend: 60% EI (exploitation) + 40% UCB (exploration) ---
    # The EI component is the primary driver; UCB provides a secondary
    # push toward uncertain regions.
    w_ei = 0.6
    w_ucb = 0.4
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
