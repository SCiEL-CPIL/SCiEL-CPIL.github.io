"""Combined Monte Carlo Expected Improvement acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Deadline guard
    start_time = time.time()
    deadline = 280.0
    
    # Clamp sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Number of samples for Monte Carlo integration of EI
    # Reduced K to increase stochasticity, which improved performance over analytical EI
    K = 20
    
    # Generate K standard normal samples for each of the n candidates
    # Shape: (K, n)
    # Using standard normal, then scaling by sigma and adding mu
    # This is equivalent to sampling from N(mu, sigma^2) at each point independently
    Z = np.random.randn(K, n)
    
    # Sampled function values: f_k(x_i)
    # Shape: (K, n)
    samples = mu[None, :] + sigma_safe[None, :] * Z
    
    # Compute improvement for each sample and candidate
    # Improvement = max(0, f_best - f_sampled)
    # Shape: (K, n)
    improvements = np.maximum(0, f_best - samples)
    
    # Average improvement over K samples
    # Shape: (n,)
    scores = np.mean(improvements, axis=0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (should be very fast, but just in case)
    if time.time() > deadline:
        # If we somehow exceeded, return a fallback (e.g., negative mu)
        # But this should not happen for K=100, n=2000
        return -mu + sigma_safe * 0.0
        
    return scores
# EVOLVE-BLOCK-END
