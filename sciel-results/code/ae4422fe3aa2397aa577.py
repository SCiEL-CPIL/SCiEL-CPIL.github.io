import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_acquisition_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using multi-sample Thompson Sampling.
    
    Instead of computing a single expected improvement score, we draw multiple
    samples from the posterior and score each candidate by how often it would
    be the best (minimum) across samples. This provides a different exploration
    mechanism than EI.
    """
    global _acquisition_start_time
    if _acquisition_start_time is None:
        _acquisition_start_time = time.time()
    
    # Deadline guard: if we're getting close to 330s, return safe scores
    elapsed = time.time() - _acquisition_start_time
    if elapsed > 325:
        # Return deterministic fallback: score by -mu (exploit best known)
        return -mu
    
    n_samples = 50
    n_candidates = len(mu)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw samples from the posterior: f(x) ~ N(mu, sigma^2)
    # samples[s, i] is sample s at candidate i
    samples = mu[np.newaxis, :] + sigma_safe[np.newaxis, :] * np.random.randn(n_samples, n_candidates)
    
    # For each sample, find the best (minimum) value
    # best_in_sample[s] = min over all candidates in sample s
    best_in_sample = np.min(samples, axis=1)  # shape (n_samples,)
    
    # For each candidate, count how many times it achieved the best value
    # We need to check: for each sample s, which candidate had the min value?
    # argmin gives the index of the min in each sample
    argmin_indices = np.argmin(samples, axis=1)  # shape (n_samples,)
    
    # Count how many times each candidate was the best
    counts = np.bincount(argmin_indices, minlength=n_candidates).astype(np.float64)
    
    # Thompson score: probability of being the best
    ts_scores = counts / n_samples
    
    # Add a small exploitation bias: candidates with lower mu get a slight boost
    # This helps when Thompson sampling is too exploratory early on
    mu_range = np.max(mu) - np.min(mu)
    if mu_range > 1e-12:
        # Normalize mu to [0, 1], lower mu = higher score
        mu_norm = (mu - np.min(mu)) / mu_range
        exploit_bias = 0.1 * (1.0 - mu_norm)
    else:
        exploit_bias = np.zeros(n_candidates)
    
    # Combine: Thompson score + exploitation bias
    final_scores = ts_scores + exploit_bias
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(final_scores)):
        final_scores = -mu
    
    return final_scores
