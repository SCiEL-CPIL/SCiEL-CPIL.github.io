import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Draw a sample from the GP posterior: x ~ N(mu, sigma^2).
       We use a deterministic seed based on f_best to ensure reproducibility 
       for the same state, but different states yield different samples.
    2. The score is the sample value divided by the cost.
       This naturally incorporates uncertainty: high sigma leads to larger 
       deviations in the sample, potentially finding better regions, while 
       dividing by cost penalizes expensive evaluations.
    
    Note: Since we don't have access to the random number generator state 
    of the main loop, we derive a seed from f_best. This makes the 
    acquisition deterministic for a given (mu, sigma, f_best, cost) input, 
    satisfying the requirement for finite scores and reproducibility.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Derive a seed from f_best to make the sampling deterministic per state.
    # f_best is a float. We need an integer seed.
    # Using a hash of the bits of f_best ensures different f_best values 
    # lead to different samples.
    seed_val = int(np.abs(f_best) * 1e6) % (2**32)
    
    # Create a random generator
    rng = np.random.RandomState(seed_val)
    
    # Draw samples from N(mu, sigma)
    # sigma is the standard deviation
    samples = rng.normal(loc=mu, scale=sigma_safe)
    
    # Cost-aware score: We want to minimize f(x)/cost.
    # For acquisition (higher is better), we maximize -f(x)/cost or similar.
    # However, Thompson Sampling typically minimizes the sample.
    # If we use score = -sample / cost, we are maximizing the negative of the 
    # expected cost-normalized function value.
    
    # Alternative view:
    # The evaluator picks the candidate with the highest score.
    # We want that candidate to be the one that minimizes f(x)/cost in expectation.
    # So score = - sample / cost_safe.
    
    score = -samples / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
