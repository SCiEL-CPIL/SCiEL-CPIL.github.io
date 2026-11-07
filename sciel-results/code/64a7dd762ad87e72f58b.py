"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time for safety
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo Expected Improvement (MCEI).
    
    Instead of using a closed-form EI or a heuristic LCB, we approximate the 
    expected improvement by drawing multiple samples from the posterior 
    distribution N(mu, sigma^2) and computing the average improvement 
    over f_best. This approach is robust, handles uncertainty naturally, 
    and avoids the pitfalls of fixed beta schedules in LCB or potential 
    local optima in deterministic EI calculations.
    
    For each of K samples:
      1. Draw z ~ N(0, 1)
      2. Compute f_sample = mu + sigma * z
      3. Compute improvement = max(0, f_best - f_sample)
    Return the mean improvement across K samples.
    
    Higher scores indicate candidates expected to improve the best value more.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a safe, neutral score if we are running out of time
        return np.zeros_like(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Number of Monte Carlo samples for EI approximation
    # K=100 provides a good balance between accuracy and speed
    K = 100
    
    # Generate random normal samples: shape (K, 2000)
    # Using standard normal distribution
    z = np.random.standard_normal((K, len(mu)))
    
    # Compute samples: f_sample = mu + sigma * z
    # mu and sigma have shape (2000,), z has shape (K, 2000)
    # Broadcasting: mu and sigma are reshaped to (1, 2000)
    mu_reshaped = mu.reshape(1, -1)
    sigma_reshaped = sigma_safe.reshape(1, -1)
    
    f_samples = mu_reshaped + sigma_reshaped * z  # Shape: (K, 2000)
    
    # Compute improvement for each sample: max(0, f_best - f_sample)
    # f_best is a scalar
    improvements = np.maximum(0, f_best - f_samples)  # Shape: (K, 2000)
    
    # Average improvement across K samples
    # Shape: (2000,)
    expected_improvement = np.mean(improvements, axis=0)
    
    # Ensure finite values
    score = np.where(np.isfinite(expected_improvement), expected_improvement, 0.0)
    
    return score
# EVOLVE-BLOCK-END
