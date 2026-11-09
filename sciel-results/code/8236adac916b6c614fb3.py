import numpy as np
import time

# Module-level state for time tracking
_start_time = None
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Draws a single sample from the GP posterior predictive distribution
    and returns the negative sampled values as scores. The candidate with
    the highest score (lowest sampled value) is selected for evaluation.
    """
    global _start_time
    
    # Initialize time tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline - if we're close to the limit, fall back to a simple
    # exploitation-based score to avoid any overhead
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Emergency fallback: just use negative mean
        return -mu
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw samples from the standard normal distribution
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute the sampled objective values
    # f_sampled ~ N(mu, sigma^2)
    sampled_f = mu + sigma_safe * z
    
    # The evaluator picks the candidate with the highest score.
    # We want to minimize the function, so we want the candidate with the
    # lowest sampled value to have the highest score.
    scores = -sampled_f
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
