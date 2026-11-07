"""Rewritten acquisition for Bayesian optimization of Hartmann-3D using model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 15.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling (single-sample approximation of Knowledge Gradient).
    This approach samples from the posterior distribution and selects the point with the lowest
    sampled value, effectively balancing exploration and exploitation by allowing high-uncertainty
    regions to occasionally yield very low (good) sampled values.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        # Fallback to EI if time is critical
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: Draw one sample from the posterior for each candidate
    # The score is the negative of the sampled value, so higher scores correspond to lower (better) objective values
    # We use a fixed seed for reproducibility within the step if needed, but standard practice is random
    # However, to avoid pathological behavior if sigma is 0, we add a tiny epsilon
    
    # Generate standard normal random numbers
    z_samples = np.random.randn(len(mu))
    
    # Sampled values
    sampled_values = mu + sigma_safe * z_samples
    
    # The acquisition score is the negative of the sampled value
    # We want to maximize the score, so we minimize the sampled value
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
