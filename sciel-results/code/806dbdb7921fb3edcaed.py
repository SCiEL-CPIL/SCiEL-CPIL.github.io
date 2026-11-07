import numpy as np
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # 300s limit, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Multi-sample Thompson Sampling.
    
    This approach reduces the variance of the single-sample Thompson Sampling
    by drawing multiple samples per candidate and taking the minimum (for minimization).
    This provides a more robust estimate of the potential improvement in high-uncertainty regions.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard: if time is running out, fall back to a fast deterministic heuristic
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        # Fallback to UCB with beta=1
        return -mu + 1.0 * sigma_safe

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Number of samples per candidate to reduce variance in the acquisition decision
    n_samples = 50
    
    # Draw samples: shape (n_samples, 2000)
    # mu and sigma_safe are broadcast to match the size
    samples = np.random.normal(loc=mu, scale=sigma_safe, size=(n_samples, len(mu)))
    
    # For minimization, we are interested in the lowest values.
    # Taking the minimum of the samples for each candidate gives a robust
    # estimate of the "best possible" value in that region.
    min_values = np.min(samples, axis=0)
    
    # Return negative so that higher score corresponds to lower function value
    return -min_values
