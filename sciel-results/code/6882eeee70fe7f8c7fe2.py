"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for adaptive exploration
_start_time = None
_call_count = 0
_exploration_weight = 0.1  # Initial exploration weight

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement with adaptive exploration.
    Uses a weighted combination of EI and sigma to balance exploitation and exploration.
    
    Args:
        mu: numpy array of shape (2000,) posterior means.
        sigma: numpy array of shape (2000,) posterior standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        numpy array of shape (2000,) scores. Higher score means higher priority.
    """
    global _start_time, _call_count, _exploration_weight
    
    # Initialize timing if not already done
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: ensure we finish well before 330s
    elapsed = time.time() - _start_time
    if elapsed > 325:  # Safety margin
        # Return a simple score to avoid timeout
        return -mu
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # Improvement = f_best - f(x), where we want to maximize this
    # For a GP with mean mu and std sigma, E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues in normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration weight: start high, decay over time
    # Early on, we want more exploration; later, more exploitation
    if _call_count <= 5:
        # First few evaluations: heavy exploration
        _exploration_weight = 0.5
    elif _call_count <= 10:
        _exploration_weight = 0.3
    elif _call_count <= 20:
        _exploration_weight = 0.15
    else:
        _exploration_weight = 0.05
    
    # Normalize sigma to have similar scale as EI
    sigma_scaled = sigma / (np.max(sigma) + 1e-10) if np.max(sigma) > 0 else sigma
    
    # Combine EI and exploration
    # Higher EI is better, higher sigma is more exploratory
    scores = ei + _exploration_weight * sigma_scaled * (np.max(ei) + 1e-10)
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
