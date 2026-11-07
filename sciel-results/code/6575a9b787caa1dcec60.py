"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for tracking call count and time
_call_count = 0
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Combines a decaying UCB strategy (strong exploration early, exploitation later)
    with an Expected Improvement component. The UCB term is scaled by a coefficient
    that decays over the course of the optimization to shift focus from exploration
    to exploitation. The EI term provides a baseline for improvement.
    
    Args:
        mu: array of shape (n,) - predicted means
        sigma: array of shape (n,) - predicted standard deviations
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (n,) - acquisition scores (higher = better)
    """
    global _call_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple EI to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Decaying UCB coefficient
    # We have 25 total evaluations, 5 initial, so 20 acquisition calls
    # Decay beta from 4.0 to 1.0 over the 20 calls
    if _call_count <= 5:
        beta = 4.0
    elif _call_count <= 10:
        beta = 3.0
    elif _call_count <= 15:
        beta = 2.0
    else:
        beta = 1.5
    
    # UCB for minimization: -mu + beta * sigma
    # We scale the UCB term to be comparable in magnitude to EI
    # EI is typically in the range of [0, f_best - min(mu)]
    # UCB is in the range of [-max(mu), -min(mu) + beta*max(sigma)]
    # We normalize both to have similar scales
    
    # Scale EI: it's already in objective units
    # Scale UCB: we want it to be in similar units
    # A simple approach: use a weighted sum where the weight of UCB decays
    
    # The key insight from the inspiration: pure UCB with decay worked well
    # Let's combine EI (for exploitation) with UCB (for exploration)
    # but weight the UCB term with a decaying coefficient
    
    # Normalize EI and UCB to be on similar scales
    # EI is non-negative, UCB can be negative
    # We'll use: score = ei + alpha * (-mu + beta * sigma)
    # where alpha decays over time
    
    # Actually, let's follow the inspiration more closely:
    # Use UCB as the primary driver, but add a small EI component
    # The UCB already captures both exploitation (low mu) and exploration (high sigma)
    
    # UCB for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine: primarily UCB, with a small EI boost for exploitation
    # The EI term helps focus on points where improvement is likely
    alpha = 0.3  # Weight for EI component
    
    scores = ucb + alpha * ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
