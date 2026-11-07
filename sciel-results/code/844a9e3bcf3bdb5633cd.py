import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pure Thompson Sampling.
    
    Samples from the GP posterior predictive distribution to select the next point.
    For minimization, we maximize the negative of a sample from N(mu, sigma^2).
    This stochastic approach naturally balances exploration and exploitation,
    which is critical for multimodal landscapes like Michalewicz 2D.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The best (lowest) objective value observed so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw z ~ N(0, 1) for each candidate
    z = np.random.standard_normal(mu.shape)
    
    # Sample from posterior: f_sample = mu + sigma * z
    # For minimization, higher score = more promising (lower objective value)
    # So we negate the sample: score = -(mu + sigma * z)
    score = -(mu + sigma_safe * z)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
