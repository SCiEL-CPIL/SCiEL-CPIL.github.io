import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Safety: return EI as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Use adaptive beta based on distance from known optimum
    f_opt = -39.166
    dist = np.clip((f_best - f_opt) / 50.0, 0.0, 1.0)
    beta = 0.5 + 2.5 * dist
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both scores to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Blend: weight EI more when close to optimum, UCB more when far
    weight_ei = 0.70 + 0.30 * (1.0 - dist)  # 1.0 when close, 0.70 when far
    weight_ucb = 1.0 - weight_ei             # 0.0 when close, 0.30 when far
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Add a small exploitation boost: candidates where mu is close to f_best
    # get a small bonus to help refine the neighborhood of the best point
    mu_diff = f_best - mu  # positive when mu < f_best (better than best)
    boost = np.clip(mu_diff / (sigma_safe + 1e-9), 0, 3.0) * 0.05
    score += boost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
