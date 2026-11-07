import numpy as np
import time

# Global deadline tracking
_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s kill

def _ensure_deadline():
    """Check if we're approaching the deadline."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return elapsed < (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-based acquisition function for minimization.
    
    The score is defined as mu - sample * sigma, where sample is drawn from
    a standard normal distribution. This approximates sampling from the
    posterior predictive distribution and selecting the candidate that
    minimizes the sampled value.
    
    For minimization, we want to find candidates where the true value is
    likely to be low. Thompson sampling works by sampling a function value
    from the posterior at each candidate point and selecting the point with
    the lowest sampled value.
    
    Since we need to return scores where higher is better, we return the
    negative of the sampled value: -(mu - z * sigma) = z * sigma - mu.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    if not _ensure_deadline():
        # Fallback: return simple EI-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-8)
        from scipy.stats import norm
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a sample from the posterior predictive
    # at each candidate point. For a GP, the posterior predictive at a
    # point is N(mu_i, sigma_i^2). We draw z_i ~ N(0, 1) and compute
    # the sampled value as mu_i + z_i * sigma_i.
    #
    # For minimization, we want the candidate with the lowest sampled value.
    # The evaluator picks the candidate with the HIGHEST score, so we return
    # the negative of the sampled value: -(mu_i + z_i * sigma_i).
    #
    # However, this means we're selecting for the most negative sampled value,
    # which corresponds to the most optimistic (lowest) prediction. This is
    # exactly Thompson sampling for minimization.
    
    # Draw a single standard normal sample for all candidates
    # Using a fixed seed per call would be deterministic, but Thompson sampling
    # should be stochastic. We use the default random state.
    z = np.random.standard_normal(len(mu))
    
    # Sampled value: mu + z * sigma
    # Score (higher is better for the evaluator): -(mu + z * sigma) = -mu - z * sigma
    scores = -mu - z * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
