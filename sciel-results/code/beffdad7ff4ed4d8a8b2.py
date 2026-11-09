"""Thompson Sampling acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    
    We sample a single realization from the Gaussian posterior for all 2000 candidates
    and return the negative sampled values as scores. This naturally balances exploration
    and exploitation without requiring manual tuning of an exploration parameter.
    """
    global _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple, fast acquisition to avoid timeout
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        # Simple EI-like score as fallback
        from scipy.stats import norm
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is at least a small positive value to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: sample from the posterior
    # For each candidate, the posterior is N(mu_i, sigma_i^2)
    # We sample one value from each posterior and return negative as score
    # (since we want to minimize, lower f is better, so higher score = lower f)
    
    # Use a single random draw for efficiency
    # Sample: f_sampled = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    f_sampled = mu + sigma_safe * z
    
    # Return negative of sampled values as scores (higher is better for selection)
    scores = -f_sampled
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
