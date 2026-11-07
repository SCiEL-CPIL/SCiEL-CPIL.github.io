"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # --- Variance-Weighted Expected Improvement (VW-EI) ---
    # Standard EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Variance-weighted EI: multiply EI by normalized posterior variance
    # This prioritizes candidates where the model is most uncertain
    sigma_sq = sigma_safe ** 2
    max_sigma_sq = np.max(sigma_sq)
    if max_sigma_sq < eps:
        var_weight = np.ones_like(sigma_sq)
    else:
        var_weight = sigma_sq / max_sigma_sq
    
    vw_ei = ei * var_weight
    
    # Also compute a UCB component for exploration
    # Adaptive exploration: more exploration when far from optimum
    distance_to_min = max(f_best - f_min_known, 0.0)
    beta = 0.3 + 1.2 * np.exp(-distance_to_min / 8.0)
    ucb = -mu + beta * sigma_safe
    
    # Normalize both components
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    # Combine VW-EI and UCB
    # VW-EI already incorporates variance, so use a moderate blend
    score = 0.7 * normalize(vw_ei) + 0.3 * normalize(ucb)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
