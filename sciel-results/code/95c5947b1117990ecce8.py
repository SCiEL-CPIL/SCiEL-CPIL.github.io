"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Thompson Sampling for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling
    # Sample from the posterior distribution N(mu, sigma^2)
    # We want to minimize the function, so we select the candidate with the lowest sample value.
    # The interface requires returning scores where HIGHER means evaluate next.
    # Therefore, we return the negative of the samples.
    
    # Generate standard normal random numbers
    # Using np.random.normal is efficient
    random_samples = np.random.normal(0, 1, size=mu.shape)
    
    # Calculate sampled function values
    samples = mu + sigma_safe * random_samples
    
    # Return negative samples so that the lowest function value corresponds to the highest score
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
