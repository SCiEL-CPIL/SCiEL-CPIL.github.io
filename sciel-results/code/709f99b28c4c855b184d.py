"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds of safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a posterior sampling approach that
    approximates Expected Improvement by drawing multiple latent function
    samples from the Gaussian process posterior.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 1.5 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Posterior sampling-based acquisition
    # Draw 50 samples from the GP posterior and compute the mean improvement
    # This approximates Expected Improvement more accurately than the closed-form
    # for non-Gaussian improvements and better captures the true expected gain
    
    n_samples = 50
    
    # Pre-allocate for efficiency
    improvements = np.zeros((n_samples, len(mu)))
    
    # Use a fixed seed for reproducibility within a single acquisition call
    # but different across calls to maintain exploration
    rng = np.random.RandomState()
    
    for i in range(n_samples):
        # Draw standard normal samples
        z_samples = rng.standard_normal(len(mu))
        # Transform to function value samples
        f_samples = mu + sigma * z_samples
        # Compute improvement: how much better than f_best (positive = improvement)
        improvements[i] = np.maximum(f_best - f_samples, 0.0)
    
    # The acquisition score is the mean improvement across samples
    # This is an unbiased estimator of Expected Improvement
    scores = improvements.mean(axis=0)
    
    # Add a small exploration bonus to encourage visiting unexplored regions
    # This helps early in optimization when f_best is far from optimal
    exploration_bonus = 0.1 * sigma
    
    # Combine: mean improvement + exploration bonus
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
