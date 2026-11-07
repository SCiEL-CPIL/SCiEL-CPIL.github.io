import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before the 330s hard limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    
    Uses Probability of Improvement (PI) with a small exploration jitter.
    PI focuses on points with a high probability of beating the current best,
    which avoids the heavy-tailed exploration of EI that can waste evaluations
    in high-dimensional spaces where the GP mean is noisy.
    
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
    if time.time() - _start_time > _TIME_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-8)
        z = (f_best - mu_arr) / sigma_safe
        pi = norm.cdf(z)
        return np.where(np.isfinite(pi), pi, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # norm.cdf(z) is Phi(z)
    pi = norm.cdf(z)
    
    # Add a small exploration jitter to break ties and encourage slight exploration
    # This helps in 6D spaces to avoid getting stuck in local optima
    exploration_weight = 1e-4
    exploration_term = exploration_weight * sigma_safe
    
    # Combine PI and exploration
    scores = pi + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
