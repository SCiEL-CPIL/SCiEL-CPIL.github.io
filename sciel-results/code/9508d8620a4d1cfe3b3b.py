"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of shape (2000,) containing posterior means.
        sigma: Array of shape (2000,) containing posterior standard deviations.
        f_best: The best observed value so far (not used directly in TS score calculation).
        
    Returns:
        Array of shape (2000,) with scores. Higher scores indicate better candidates.
    """
    # Thompson Sampling: Draw a sample from the posterior predictive distribution
    # f(x) ~ N(mu(x), sigma(x)^2)
    # Since we are minimizing, we want to find the x that minimizes the sampled value.
    # The evaluator selects the candidate with the HIGHEST score.
    # So we return -sampled_value.
    
    # Generate standard normal samples
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute the sampled function values
    # Using mu - sigma * z ensures that if we want to minimize, lower samples are better.
    # Actually, f ~ N(mu, sigma^2) means f = mu + sigma * z.
    # We want to minimize f.
    # The score should be high for low f.
    # So score = -f = -(mu + sigma * z).
    # Alternatively, we can think of maximizing -f.
    
    sampled_values = mu + sigma * z
    
    # Return negative of sampled values so that higher score corresponds to lower function value
    return -sampled_values
# EVOLVE-BLOCK-END
