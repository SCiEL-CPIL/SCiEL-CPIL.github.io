"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
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
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling Strategy:
    # 1. Sample a random realization of the objective function from the GP posterior.
    #    f_sample(x) ~ N(mu(x), sigma(x)^2)
    #    f_sample = mu + sigma * z, where z ~ N(0, 1)
    
    # Use a fixed seed for reproducibility of the random draw in this specific call?
    # No, Thompson sampling typically uses fresh randomness each step to explore.
    # However, we need to return scores.
    # The "score" for a candidate x in Thompson sampling is effectively the negative of the sampled value,
    # because we want to pick the x with the lowest sampled value (minimization).
    # So, score(x) = -f_sample(x) = -(mu(x) + sigma(x) * z)
    
    # Generate a single random normal variable z.
    # We use np.random.normal(0, 1). 
    # Note: The evaluator calls this function 25 times per seed (30 total evals - 5 initial).
    # The randomness should be fresh each time to allow exploration.
    
    z = np.random.normal(0.0, 1.0)
    
    # Calculate the sampled function values
    # f_sample = mu + sigma * z
    f_sample = mu + sigma_safe * z
    
    # The acquisition score should be high for good candidates.
    # Since we minimize f, a low f_sample is good.
    # So score = -f_sample
    scores = -f_sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
