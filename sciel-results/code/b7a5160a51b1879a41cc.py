"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function.
    
    Instead of computing a deterministic score (like EI or LCB) for each candidate,
    this strategy simulates a single sample from the posterior distribution of the
    objective function values at the candidate points.
    
    Since we do not have access to the full covariance matrix or the exact GP
    hyperparameters in the acquisition function (only mu and sigma), we approximate
    the posterior sample using a linear Gaussian assumption:
    
    f_sample = mu + sigma * z
    
    where z is a standard normal random vector.
    
    To make this a valid "acquisition" that returns scores, we return the negative
    of the sampled values. Maximizing the returned score is equivalent to minimizing
    the sampled function value. This stochastic nature helps escape local optima
    and explores the search space more effectively than greedy exploitation,
    especially in the early stages of optimization.
    
    To ensure stability and avoid extreme outliers dominating the selection in a
    single shot, we use a fixed seed per call? No, the task says "draws 2000 fresh
    uniform candidates". It doesn't restrict random state. However, using a random
    sample every time is the essence of Thompson Sampling.
    
    One potential issue: If sigma is 0, the sample is just mu.
    If sigma is large, the sample is noisy.
    
    We will generate a single random vector z ~ N(0, I) of size 2000.
    Then score = -(mu + sigma * z).
    
    This is computationally O(N) and very fast.
    """
    
    # Deadline guard
    start_time = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge case where sigma might be negative (shouldn't happen, but safe)
    sigma = np.maximum(sigma, 0.0)
    
    # Generate standard normal samples
    # Using np.random.randn is fast. 
    # We don't need a specific seed to be passed in, so we use the global numpy random state.
    # This ensures different exploration at each step.
    z = np.random.randn(mu.shape[0])
    
    # Compute the sampled function values
    # f_sample = mu + sigma * z
    # We want to minimize f, so we maximize -f_sample
    sample_values = mu + sigma * z
    
    # Return negative sampled values as scores
    scores = -sample_values
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline (unlikely to be hit, but required by directive)
    if time.time() - start_time > 290.0:
         # Fallback to pure exploitation if somehow slow
         return -mu
         
    return scores
# EVOLVE-BLOCK-END
