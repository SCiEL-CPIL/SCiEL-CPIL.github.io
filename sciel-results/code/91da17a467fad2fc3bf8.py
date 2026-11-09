"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce deadline
_START_TIME = time.time()
_DEADLINE = 300.0  # 300 seconds wall-clock limit

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (N,) - mean of the GP posterior
    sigma: array of shape (N,) - standard deviation of the GP posterior
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (N,) - scores for each candidate (higher is better)
    """
    # Check deadline
    current_time = time.time()
    if current_time > _DEADLINE - 5.0:
        # If we are close to the deadline, return a deterministic fallback
        # to ensure we don't time out. 
        # A safe fallback is to exploit the current best mean.
        # Since we want to minimize, lower mu is better.
        # But we must return finite scores.
        # Using -mu ensures that points with lower predicted mean get higher score.
        # This is a pure exploitation strategy, which is safer than random sampling
        # if we are running out of time.
        return -mu
    
    # Thompson Sampling:
    # For minimization, we want to find the point with the lowest value.
    # We sample from the posterior N(mu, sigma^2) for each point.
    # The candidate with the lowest sample is the most promising.
    # To convert this to a "higher is better" score:
    # We can return the negative of the sample.
    # However, standard Thompson Sampling usually picks the argmin of the samples.
    # The evaluator picks the argmax of the returned scores.
    # So if we return -sample, the argmax of -sample is the argmin of sample.
    
    # Generate random normal noise
    # Using a fixed seed per call is not possible as we don't have access to global state easily
    # but np.random is fine. Note: The evaluator runs 3 seeds.
    # To ensure reproducibility within a seed run if the function is called multiple times?
    # The task says "at each step ... calls acquisition".
    # Thompson sampling relies on the randomness of the sample.
    
    # One issue: If we just return -sample, the scores are random every time.
    # This is correct for TS.
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Sample from N(mu, sigma^2)
    # Note: In Python, np.random.normal(loc, scale)
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # We want to minimize the function value.
    # The "best" candidate is the one with the lowest sample.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    # Ensure finite values
    # np.random.normal can theoretically produce inf if scale is huge, but sigma is from GP.
    # Clip to be safe.
    scores = np.clip(scores, -1e10, 1e10)
    
    return scores
# EVOLVE-BLOCK-END
