"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition.
    
    Instead of returning a deterministic score, we sample from the GP posterior
    for each candidate and return the negative of the sample (since higher scores
    are preferred by the evaluator, and we want to minimize the objective).
    
    mu: shape (2000,) - mean of GP posterior at candidates
    sigma: shape (2000,) - std of GP posterior at candidates
    f_best: float - best observed value so far
    
    Returns: shape (2000,) - scores where higher means evaluate next.
             We sample y_i ~ N(mu_i, sigma_i^2) and return -y_i so that
             candidates with low sampled objective values get high scores.
    """
    # Sample from the posterior: y_i ~ N(mu_i, sigma_i)
    # Use a fixed seed per call? No, we want stochasticity.
    # But we need reproducibility within a single acquisition call.
    # The evaluator calls this function and picks argmax.
    # We just need to return finite scores.
    
    # Handle zero or near-zero sigma to avoid issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Draw standard normal samples
    eps = np.random.standard_normal(mu.shape)
    
    # Sampled objective values
    samples = mu + sigma_safe * eps
    
    # Return negative samples so that lower objective values -> higher scores
    return -samples
# EVOLVE-BLOCK-END
