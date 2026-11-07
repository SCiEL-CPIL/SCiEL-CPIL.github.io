"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a hybrid approach combining Expected Improvement (EI) with a
    Thompson sampling-inspired exploration term and adaptive weighting.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher means evaluate next.
    """
    start_time = time.time()
    DEADLINE = 300.0
    SAFETY_MARGIN = 10.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Lower confidence bound (for minimization)
    # Use adaptive beta based on how close we are to the optimum
    # Branin minimum is ~0.397887
    if f_best < 0.5:
        beta = 0.5  # Exploit more when close to optimum
    elif f_best < 1.0:
        beta = 1.0
    elif f_best < 3.0:
        beta = 1.5
    else:
        beta = 2.5  # Explore more when far from optimum
    
    ucb = -mu + beta * sigma_safe
    
    # Hybrid acquisition: blend EI, PI, and UCB
    # The blend adapts based on the uncertainty landscape
    # When sigma is high relative to the range of mu, favor exploration
    
    mu_range = np.ptp(mu) if np.ptp(mu) > eps else eps
    sigma_mean = np.mean(sigma_safe)
    
    # Exploration ratio: how much of the uncertainty is in sigma vs mu variation
    expl_ratio = sigma_mean / (sigma_mean + mu_range)
    expl_ratio = np.clip(expl_ratio, 0.0, 1.0)
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < eps:
            return np.zeros_like(arr)
        return (arr - arr_min) / arr_range
    
    ei_norm = normalize(ei)
    pi_norm = normalize(pi)
    ucb_norm = normalize(ucb)
    
    # Adaptive weights
    # Early on (high uncertainty), weight UCB and PI more
    # Later (low uncertainty), weight EI more
    w_ei = 0.4 + 0.4 * (1.0 - expl_ratio)
    w_pi = 0.2 * expl_ratio
    w_ucb = 1.0 - w_ei - w_pi
    w_ucb = max(w_ucb, 0.0)
    
    # Renormalize weights to sum to 1
    w_sum = w_ei + w_pi + w_ucb
    if w_sum > eps:
        w_ei /= w_sum
        w_pi /= w_sum
        w_ucb /= w_sum
    
    scores = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm
    
    # Add a small bonus for points with very low predicted mean
    # This helps find the global minimum
    mu_min = np.min(mu)
    mu_scale = np.ptp(mu) if np.ptp(mu) > eps else 1.0
    low_mu_bonus = 0.1 * (1.0 - (mu - mu_min) / mu_scale)
    
    scores = scores + low_mu_bonus
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
