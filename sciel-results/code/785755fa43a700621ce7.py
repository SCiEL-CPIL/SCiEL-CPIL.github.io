"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Thompson Sampling:
    # We draw a sample from the posterior distribution of the function values.
    # f(x) ~ N(mu(x), sigma(x)^2)
    # To maximize the score (higher is better), we want the sample to be as low as possible.
    # So we return -sample as the score.
    
    # Ensure sigma is non-negative and has a small floor to avoid issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a random sample from the normal distribution
    # Using standard normal and scaling
    z = np.random.randn(len(mu))
    sample = mu + sigma_safe * z
    
    # The score is the negative of the sampled value.
    # Lower sampled value -> higher score.
    score = -sample
    
    # Ensure finite scores
    # If any values are non-finite (shouldn't happen with standard normal and finite inputs), replace with very low score
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
