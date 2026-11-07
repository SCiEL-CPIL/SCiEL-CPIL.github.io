"""Rewritten program using Thompson Sampling with selection probability."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition using selection probability.
    
    We sample multiple latent function values from the GP posterior and
    compute the probability that each candidate is the global minimum
    across all candidates. This better approximates the Thompson Sampling
    policy of 'pick the best point from the sampled function'.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): The best observed function value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates, shape (2000,)
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: use LCB with a reasonable beta
        beta = 2.0
        scores = -mu + beta * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e6, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Number of samples for Thompson Sampling
    # Use fewer samples to stay within time budget
    n_samples = 50
    
    # Sample latent function values: f ~ N(mu, sigma^2)
    # Shape: (n_samples, n)
    samples = mu[None, :] + sigma[None, :] * np.random.randn(n_samples, n)
    
    # For each sample, find which candidate is the global minimum
    # min_indices[i] = index of the minimum in sample i
    min_indices = np.argmin(samples, axis=1)
    
    # Count how many times each candidate is selected as the global minimum
    # This is the selection probability (unnormalized)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(counts, min_indices, 1.0)
    
    # Normalize to get probabilities
    total = counts.sum()
    if total > 0:
        scores = counts / total
    else:
        # Fallback: use LCB
        beta = 2.0
        scores = -mu + beta * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
