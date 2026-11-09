"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Structural Change: Thompson Sampling
    # Instead of computing a deterministic score like EI or LCB, we sample
    # from the posterior distribution for each candidate and select the one
    # with the lowest sampled value (since we are minimizing).
    # This provides a natural, parameter-free balance between exploration
    # (high sigma regions) and exploitation (low mu regions).
    
    # Check if we are in a highly converged state (low uncertainty)
    # If so, Thompson Sampling might be too noisy, so we fall back to EI.
    max_sigma = np.max(sigma_safe)
    
    if max_sigma < 1e-4:
        # Converged state: Use Expected Improvement for precision
        z = (f_best_safe - mu_safe) / sigma_safe
        # Clip z for numerical stability
        z = np.clip(z, -10, 10)
        ei_scores = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        final_scores = ei_scores
    else:
        # Exploration/Exploitation state: Use Thompson Sampling
        # Draw standard normal samples
        # We clip the samples to [-3, 3] to prevent extreme outliers 
        # from dominating the selection, which can happen with pure TS
        # in multi-modal landscapes if a rare large negative noise hits 
        # a moderate-mean point.
        z_samples = np.random.randn(2000)
        z_samples = np.clip(z_samples, -3.0, 3.0)
        
        # Sampled objective values
        sampled_f = mu_safe + sigma_safe * z_samples
        
        # We want to minimize f, so higher score is better for lower f
        final_scores = -sampled_f
        
        # Optional: If the best sampled value is much worse than f_best,
        # it might be a good idea to blend with EI? 
        # No, pure TS is the structural change.
        
    # Check time
    if time.time() - start_time > deadline - start_time:
        # Fallback to EI if TS somehow takes too long (unlikely, but safe)
        z = (f_best_safe - mu_safe) / sigma_safe
        z = np.clip(z, -10, 10)
        ei_scores = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        final_scores = ei_scores

    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
