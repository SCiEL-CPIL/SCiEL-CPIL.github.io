"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track state across calls within a single seed
_state = {"call_count": 0, "start_time": None}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Combines Expected Improvement (EI) with an adaptive exploration bonus.
    
    Uses a call counter to track progress within each seed (25 steps per seed).
    The exploration weight decays over time, shifting from exploration to exploitation.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    global _state
    
    # Initialize state if needed
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    _state["call_count"] += 1
    call_count = _state["call_count"]
    
    # Deadline guard: ensure we don't exceed time budget
    elapsed = time.time() - _state["start_time"]
    if elapsed > 320.0:  # Safety margin before 330s limit
        # Return a simple exploitation-based score as fallback
        sigma_safe = np.maximum(sigma, 1e-12)
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Determine step within current seed
    # 25 steps per seed (5 initial + 20 optimization)
    step = (call_count - 1) % 25
    progress = step / 25.0
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Adaptive exploration weight: decays from 2.0 to 0.5 over the seed
    # This encourages exploration early and exploitation later
    lambda_explore = 2.0 * (1.0 - progress) ** 1.5 + 0.5
    
    # Combine EI with exploration bonus
    score = ei + lambda_explore * sigma_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
