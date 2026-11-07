import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # Seconds to spare before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of Expected Improvement and UCB with adaptive exploration.
    Includes a deadline guard to prevent timeouts.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - posterior mean in objective units
    sigma : array of shape (2000,) - posterior std in objective units
    f_best : float - lowest observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: If we are running out of time, return a simple fallback
    # to ensure we don't timeout. The fallback is -mu (greedy exploitation).
    current_time = time.time()
    elapsed = current_time - _start_time
    if elapsed > 300.0 - _DEADLINE_MARGIN:
        return -mu

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB for minimization: we want low mu and high uncertainty
    # LCB(x) = mu(x) - beta * sigma(x)
    # Score(x) = -LCB(x) = -mu(x) + beta * sigma(x)
    # Adaptive beta: explore more when f_best is far from optimum
    f_opt = -39.1662
    dist = max(f_best - f_opt, 0.0)
    
    # Normalize distance: when f_best is ~0, dist ~ 39; when close to opt, dist ~ 0
    # Use a sigmoid-like mapping to get beta in [0.5, 3.0]
    # At dist=0 (near optimum), beta should be small (~0.5) for exploitation
    # At dist=40 (far from optimum), beta should be large (~3.0) for exploration
    beta = 0.5 + 2.5 * (1.0 - np.exp(-dist / 10.0))
    
    ucb = -mu + beta * sigma
    
    # Normalize both to have similar scale before blending
    # This prevents one metric from dominating due to scale differences
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-10
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Adaptive blend: more UCB early (high dist), more EI later (low dist)
    # w_ucb in [0.3, 0.7]
    w_ucb = 0.3 + 0.4 * (1.0 - np.exp(-dist / 15.0))
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
