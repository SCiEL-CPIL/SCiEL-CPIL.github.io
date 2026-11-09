"""Rewritten program using Thompson Sampling for Bayesian optimization acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global state
_global_start_time = None
_eval_count = 0
_TIMEOUT_MARGIN = 10.0
_TOTAL_LIMIT = 330.0


def _check_deadline():
    """Check if we are approaching the timeout limit."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    elapsed = time.time() - _global_start_time
    return elapsed > (_TOTAL_LIMIT - _TIMEOUT_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _global_start_time, _eval_count
    
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Increment evaluation counter
    _eval_count += 1
    
    # Check deadline - if close to timeout, return a safe deterministic fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        # Return negative Expected Improvement (since we want higher score for better candidate)
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return -ei
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Thompson Sampling: Draw a sample from the posterior at each point
    # The posterior is Gaussian: N(mu, sigma^2)
    # We sample: s = mu + sigma * z, where z ~ N(0, 1)
    
    # Generate standard normal random numbers
    # Using a fixed seed for reproducibility within the run? No, BO requires stochasticity.
    # But we need to ensure we don't use the same random state if called multiple times?
    # The evaluator calls this function once per step.
    z = np.random.standard_normal(len(mu))
    
    # Sample the function value
    samples = mu + sigma_safe * z
    
    # We want to minimize the function value.
    # The acquisition function returns scores where HIGHER is better.
    # Therefore, we return the negative of the sampled values.
    # The evaluator will pick argmax(score), which corresponds to argmin(sample).
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
