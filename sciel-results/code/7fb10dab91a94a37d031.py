"""Thompson Sampling Min-Value Acquisition for Bayesian Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Deadline guard
    start_time = time.time()
    deadline = 280.0
    
    # Clamp sigma to avoid zero variance issues in sampling
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Number of samples for Thompson Sampling
    # Taking the minimum of K samples acts as an optimistic lower bound,
    # encouraging exploration in high-uncertainty regions.
    K = 20
    
    # Generate K standard normal samples for each of the n candidates
    # Shape: (K, n)
    # We assume independent samples at each candidate point as we only have
    # marginal means and variances.
    Z = np.random.randn(K, n)
    
    # Sampled function values: f_k(x_i)
    # Shape: (K, n)
    samples = mu[None, :] + sigma_safe[None, :] * Z
    
    # Compute the minimum sampled value for each candidate
    # Shape: (n,)
    # This represents an optimistic estimate of the function value
    min_samples = np.min(samples, axis=0)
    
    # The score should be higher for better (lower) function values
    # So we negate the minimum sampled value
    scores = -min_samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (should be very fast, but just in case)
    if time.time() > deadline:
        # If we somehow exceeded, return a fallback
        # Using negative mu as a heuristic for "likely good" points
        return -mu
        
    return scores
# EVOLVE-BLOCK-END
