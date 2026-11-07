"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of calculating Expected Improvement (EI) or Upper Confidence Bound (UCB),
    this strategy samples a single value from the Gaussian posterior distribution
    N(mu, sigma^2) for each candidate. The candidate with the lowest sampled value
    is selected for evaluation. This is equivalent to returning the negative of the
    sampled values as the acquisition score (since higher scores are preferred).
    
    This approach naturally balances exploration (high sigma leads to wider spread
    of samples, increasing chance of finding a low outlier) and exploitation
    (low sigma keeps samples close to the mean, favoring known good regions).
    """
    t_start = time.time()
    deadline = 300.0  # Safety margin for the 330s kill switch
    
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases: if inputs are not finite, return uniform scores
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.ones_like(mu)
    
    # Clip sigma to avoid numerical issues with zero or negative variance
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate a single standard normal random variable
    # This is the core of Thompson Sampling: we draw one sample from the posterior
    # for all candidates simultaneously.
    z = np.random.standard_normal()
    
    # Calculate the sampled function values: f_sampled = mu + sigma * z
    # We want to minimize the function, so we want the candidate with the lowest f_sampled.
    # The acquisition score should be "higher is better", so we return -f_sampled.
    scores = -(mu + sigma_safe * z)
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time (though this operation is extremely fast, O(N))
    elapsed = time.time() - t_start
    if elapsed > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
