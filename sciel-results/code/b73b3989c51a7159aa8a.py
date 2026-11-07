# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses Thompson Sampling:
    1. Draw a sample from the GP posterior for each candidate.
    2. Select the candidate with the lowest sampled value (minimization).
    3. Return negative sampled values as scores so that argmax(scores) 
       selects the argmin(sampled_values).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
        Here, score = -sampled_value.
    """
    t_start = time.time()
    deadline = 330.0 # Seconds, with safety margin
    
    # Check deadline immediately
    if time.time() - t_start > deadline:
        # Fallback: return zeros or exploit best known mean
        return -mu

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)

    # Thompson Sampling:
    # Draw one sample from N(mu, sigma) for each candidate
    # np.random.normal(loc, scale, size)
    # We use a fixed random state? No, the evaluator likely handles global state or we just use default.
    # Using default numpy random is standard for BO unless reproducibility of the *acquisition* itself is required across runs, 
    # but the task says "evaluator starts from 5 uniform random points", implying the randomness is external or seeded by evaluator.
    # We just draw from the standard normal.
    
    samples = np.random.normal(loc=mu_safe, scale=sigma_safe)
    
    # We want to minimize the function value.
    # The candidate with the lowest sampled value is the most promising.
    # The evaluator picks the index of the MAX score.
    # So we return -samples.
    
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
