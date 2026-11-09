# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 300.0  # Safety margin before the 330s kill limit
_call_count = 0
_total_evals = 30  # 5 initial + 25 optimization steps


def acquisition(mu, sigma, f_best):
    """
    Hybrid Expected Improvement (EI) with adaptive exploration bonus.
    
    This structural change moves from an adaptive LCB to an EI-based acquisition
    function. EI directly optimizes for improvement over the current best, which
    is often more effective for multi-modal functions like Hartmann 6D when
    combined with a decaying exploration term to escape local minima.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    global _call_count
    _call_count += 1
    
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Exploration Bonus
    # We want to explore more when we are early in the search or when f_best is poor.
    # Progress: 0 at start, 1 at end.
    opt_steps = max(_call_count - 5, 1)
    total_opt = 25
    step_progress = min(opt_steps / total_opt, 1.0)
    
    # Decay exploration weight from 1.0 to 0.1
    # This ensures strong uncertainty-driven exploration early on
    exploration_weight = 0.1 + 0.9 * (1.0 - step_progress)
    
    # Combine EI and Exploration
    # The exploration term is simply the standard deviation, scaled by the weight.
    scores = ei + exploration_weight * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
