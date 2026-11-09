"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: Predictive mean of the GP at candidate points, shape (N,).
        sigma: Predictive standard deviation of the GP at candidate points, shape (N,).
        f_best: The lowest observed value so far.
        
    Returns:
        A numpy array of shape (N,) with scores. Higher scores indicate
        candidates that should be evaluated next.
    """
    # Ensure sigma is non-negative to avoid issues with sqrt if it were variance
    # Here sigma is already std dev.
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw samples from the posterior distribution N(mu, sigma^2)
    # We use standard normal distribution and scale
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma * z
    
    # The goal is to minimize the objective function.
    # Thompson Sampling selects the point with the smallest sampled value.
    # The evaluator selects the point with the HIGHEST acquisition score.
    # Therefore, we return the negative of the samples.
    return -samples
# EVOLVE-BLOCK-END
