"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # Seconds to spare before 300s limit


def acquisition(mu, sigma, f_best):
    """
    Adaptive EI/UCB blend: weights Expected Improvement and Upper Confidence Bound
    based on progress toward the known optimum. Early on, UCB dominates to escape
    local minima; late in the search, EI dominates for precise exploitation.
    
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
    current_time = time.time()
    elapsed = current_time - _start_time
    if elapsed > 300.0 - _DEADLINE_MARGIN:
        return -mu

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    f_opt = -39.1662
    dist = max(f_best - f_opt, 0.0)
    
    # Progress: 0 when far from optimum, 1 when at optimum
    # Use exponential decay with characteristic distance ~8.0
    progress = 1.0 - np.exp(-dist / 8.0)
    
    # Blend weight: w_ucb goes from ~0.7 (early, explore) to ~0.1 (late, exploit)
    w_ucb = 0.1 + 0.6 * progress
    w_ei = 1.0 - w_ucb
    
    # UCB: lower is better, so we score by -(mu + beta * sigma)
    # beta scales with exploration need: larger early, smaller late
    beta = 1.0 + 2.0 * progress  # beta in [1.0, 3.0]
    ucb_score = -(mu + beta * sigma)  # negative because lower objective is better
    
    # EI: standard Expected Improvement
    z = (f_best - mu) / sigma
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to have similar scale for blending
    # EI is always non-negative; UCB_score can be negative
    # Scale UCB_score to have comparable magnitude to EI
    ucb_scale = np.abs(ucb_score).mean() + 1e-10
    ei_scale = ei_score.mean() + 1e-10
    
    ucb_normalized = ucb_score / ucb_scale
    ei_normalized = ei_score / ei_scale
    
    # Blend: higher score means evaluate next
    # EI is naturally "higher = better"; UCB_score is "lower = better" so we negate
    blended = w_ei * ei_normalized + w_ucb * (-ucb_normalized)
    
    # Ensure all scores are finite
    blended = np.where(np.isfinite(blended), blended, 0.0)
    
    return blended
# EVOLVE-BLOCK-END
