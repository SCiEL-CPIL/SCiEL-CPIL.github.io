import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before the 330s hard limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    UCB balances exploitation (low mu) and exploration (high sigma).
    In high-dimensional multimodal landscapes like Hartmann 6D, UCB often
    outperforms EI by avoiding premature convergence to local optima.
    
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
    global _start_time
    
    # Deadline guard: If we are running out of time, return a safe, valid score.
    # Returning pure UCB is a safe fallback.
    if time.time() - _start_time > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-8)
        scores = mu_arr - 1.0 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero (though UCB doesn't divide)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB: mu - beta * sigma
    # We choose beta = 1.0 to encourage exploration in the high-dimensional space.
    beta = 1.0
    scores = mu - beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
