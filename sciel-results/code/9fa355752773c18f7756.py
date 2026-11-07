"""Rewritten acquisition for BO of a fixed seeded RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Keep a module-level clock for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 325.0  # 5s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: returns 2000 finite scores (higher = evaluate next).
    Uses a combination of Expected Improvement and UCB with adaptive exploration.
    """
    # Deadline guard: if we're running out of time, fall back to simple EI
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Emergency fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Compute UCB with adaptive beta
    # Start with a moderate exploration rate that decays
    # We don't know the step number, so use a heuristic based on the spread of sigma
    # If sigma is large relative to the range of f_best - mu, we're early in optimization
    # Use a fixed beta that balances exploitation and exploration well for this problem
    
    # Adaptive beta: scale with the median sigma to normalize
    med_sigma = np.median(sigma)
    if med_sigma > 1e-8:
        # Normalize EI and UCB to comparable scales
        # Use a beta that encourages exploration when uncertainty is high
        beta = 2.0 * med_sigma
        ucb = mu + beta * sigma_safe
        # We want to minimize, so lower mu + beta*sigma is better
        # Convert to "higher is better" score
        ucb_score = -ucb
        
        # Normalize both to [0, 1] range for blending
        ei_min, ei_max = np.min(ei), np.max(ei)
        ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
        
        if ei_max - ei_min > 1e-12:
            ei_norm = (ei - ei_min) / (ei_max - ei_min)
        else:
            ei_norm = np.zeros_like(ei)
            
        if ucb_max - ucb_min > 1e-12:
            ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
        else:
            ucb_norm = np.zeros_like(ucb_score)
        
        # Blend: weight EI more heavily as optimization progresses
        # Use the fraction of points with significant uncertainty as a proxy
        high_uncertainty_frac = np.mean(sigma_safe > 0.1 * (f_best - np.min(mu) + 1e-9))
        # If high uncertainty fraction is high, we're early -> explore more
        w_explore = min(0.5, high_uncertainty_frac * 0.5)
        
        score = (1 - w_explore) * ei_norm + w_explore * ucb_norm
    else:
        # Very low uncertainty, just use EI
        score = ei
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
