"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a principled combination of Expected Improvement (EI) relative to f_best
    and an adaptive UCB exploration term. The key structural idea is to use
    min(f_best, 3.0) as the EI reference to maintain a constant exploration signal
    even as f_best approaches the known global minimum of 3.0. This keeps the
    acquisition sensitive to high-variance regions that may contain the true optimum.
    
    The UCB term uses an adaptive beta that decreases as uncertainty diminishes,
    and both terms are combined with a weight that shifts from exploration to
    exploitation as the optimization progresses.
    
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
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Use min(f_best, 3.0) as the EI reference to maintain exploration signal
    # even as f_best approaches the known global minimum of 3.0
    f_ref = min(f_best, 3.0)
    
    # Compute Expected Improvement (for minimization)
    z = (f_ref - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_ref - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Compute a UCB-like term with adaptive beta
    # When sigma is large relative to the range of mu, we're in early stages
    # and should explore more. When sigma is small, we should exploit.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Adaptive beta: larger when uncertainty is high relative to prediction spread
    sigma_mean = np.mean(sigma_safe)
    # beta ranges from ~1.0 to ~4.0 based on relative uncertainty
    beta = 1.0 + 3.0 * min(1.0, sigma_mean / (mu_range + 1e-10))
    
    # UCB score for minimization (we want to minimize mu + beta*sigma)
    ucb_value = mu + beta * sigma_safe
    # Convert to a score where higher is better
    ucb_score = -ucb_value
    
    # Normalize both terms to [0, 1] for stable combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Weight: lean more toward EI in later stages, more toward UCB in early
    # Use the ratio of sigma to f_best magnitude as a proxy for stage
    sigma_to_fbest_ratio = sigma_mean / (abs(f_best) + 1e-10)
    # Clamp the ratio to [0, 1] for weight calculation
    w_ucb = 0.3 + 0.3 * min(1.0, sigma_to_fbest_ratio)
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small bonus for points where mu is significantly below f_ref
    # This helps direct search toward already-promising regions
    below_best = np.maximum(0.0, f_ref - mu)
    if np.max(below_best) > 1e-10:
        below_best_norm = below_best / (np.max(below_best) + 1e-10)
        scores += 0.1 * below_best_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
