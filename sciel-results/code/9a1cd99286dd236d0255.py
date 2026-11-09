"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer to track elapsed time for the deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds, as per task description

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,), posterior mean
        sigma: array of shape (2000,), posterior standard deviation
        f_best: float, best observed value so far
    
    Returns:
        array of shape (2000,), acquisition scores (higher = better)
    """
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    # This ensures we don't timeout. Thompson Sampling is very fast, so this 
    # branch is unlikely to be taken, but it satisfies the structural requirement.
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        # Fallback: Simple UCB-like score, deterministic and fast
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe

    # Ensure sigma is positive for sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling:
    # Draw a sample from the posterior Gaussian distribution at each candidate point.
    # y_i ~ N(mu_i, sigma_i^2)
    # We want to minimize f, so we select the point with the lowest sampled value y.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return score = -y.
    
    # Generate standard normal random variables
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample the function values
    y = mu + sigma_safe * z
    
    # Return negative sampled values so that lower function values get higher scores
    scores = -y
    
    # Ensure finite values to avoid invalid results
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
