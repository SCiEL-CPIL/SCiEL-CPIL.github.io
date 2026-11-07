import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard and step tracking
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before the 330s hard limit
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an adaptive exploration bonus.
    
    Uses Gaussian EI for exploitation and adds a weighted sigma term for exploration.
    The exploration weight decays over time to shift from exploration to exploitation.
    This balances finding new minima with exploring uncertain regions in the 6D landscape.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    global _start_time, _call_count
    
    # Deadline guard: If we are running out of time, return a safe, valid score.
    # Returning pure EI (without exploration) is a safe fallback.
    if time.time() - _start_time > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-8)
        z = (f_best - mu_arr) / sigma_safe
        ei = (f_best - mu_arr) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight: starts high, decays over time
    # Total evaluations: 5 initial + 25 optimization = 30
    # We want more exploration early, less later
    step = _call_count  # 0 for first call after initial 5, so 1 for first optimization step
    # Decay: start at 0.3, decay to 0.05 over 25 steps
    exploration_weight = 0.3 - 0.25 * (step / 25.0)
    exploration_weight = max(exploration_weight, 0.05)
    
    exploration_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    _call_count += 1
    
    return scores
