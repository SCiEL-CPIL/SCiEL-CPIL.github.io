"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a hybrid approach combining Expected Improvement (EI) with
    Upper Confidence Bound (UCB) exploration. The key structural idea
    is to use UCB's explicit exploration term (-mu + beta*sigma) which
    scored well (1.42) on this problem, while adding a small EI component
    to maintain exploitation capability.
    
    For Branin-2D's multimodal landscape, pure UCB with beta=1.0 performed
    best. We use that as the primary signal with a small EI refinement.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5 second safety margin before 330s limit
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Primary signal: UCB for minimization
    # Higher score = more promising (lower predicted mean, higher uncertainty)
    # score_ucb = -mu + beta * sigma
    beta = 1.0
    ucb_score = -mu + beta * sigma_safe
    
    # Secondary signal: Expected Improvement
    # EI rewards points where we expect improvement over f_best
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be comparable with UCB scale
    # EI is typically much smaller in magnitude than UCB
    # Scale EI by a small factor to add refinement without dominating
    ei_weight = 0.1
    score = ucb_score + ei_weight * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
