import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before the 330s hard limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    
    Combines Expected Improvement (EI) with a small exploration term.
    EI focuses on points likely to improve upon f_best.
    The exploration term (0.1 * sigma) encourages visiting uncertain regions
    to help escape local minima in the 6D landscape.
    
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
    
    # Add a small exploration term to balance exploitation and exploration.
    # This helps in 6D spaces to avoid getting stuck in local optima.
    exploration_weight = 0.1
    exploration_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
