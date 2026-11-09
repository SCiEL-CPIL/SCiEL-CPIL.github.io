import numpy as np
import time

# Global start time to track elapsed time if needed
_start_time = time.time()
_TIMEOUT = 330.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Returns scores such that the candidate with the highest score is selected.
    Since we are minimizing, we sample from the posterior N(mu, sigma^2)
    and return the negative of the sample.
    """
    # Check time
    if time.time() - _start_time > _TIMEOUT - _MARGIN:
        # Fallback to EI if we are running out of time (unlikely)
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
    try:
        # Ensure sigma is positive
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # Sample from the posterior
        # Shape of mu and sigma is (2000,)
        samples = mu + sigma_safe * np.random.randn(mu.shape[0])
        
        # We want to minimize f.
        # The evaluator picks the max score.
        # So we return -samples.
        scores = -samples
        
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        return scores
    except Exception:
        # Fallback to a simple heuristic if something goes wrong
        # Return -mu to exploit current best estimates
        return -mu
