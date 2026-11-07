"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track evaluation count and start time for deadline guard
_eval_count = 0
_start_time = None
_rng = np.random.default_rng(42)

def acquisition(mu, sigma, f_best):
    global _eval_count
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
        
    _eval_count += 1
    
    # Deadline guard: The evaluator kills after 330s.
    # We have 30 evaluations. If we are close to the deadline, 
    # we should return a safe, fast result. 
    # However, the acquisition function itself is called 30 times. 
    # Each call must be fast. 
    # Thompson Sampling on 2000 points is O(N), so it's very fast.
    # No complex iterative optimization is performed here, so timeout risk is low.
    
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=-1e-9)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling Strategy:
    # 1. Draw a single sample from the posterior GP at the candidate points.
    #    sample = mu + sigma * z, where z ~ N(0, 1)
    # 2. Select the candidate with the minimum sampled value.
    # 3. Return scores such that the selected candidate has the highest score.
    #    Since we want to minimize f, the "best" sample has the lowest value.
    #    We return scores = -sampled_values.
    #    The candidate with the lowest sampled value will have the highest score.
    
    # Draw standard normal noise for all candidates
    z = _rng.standard_normal(size=mu.shape)
    
    # Compute the sampled function values
    sampled_values = mu + sigma * z
    
    # Ensure finite sampled values
    sampled_values = np.nan_to_num(sampled_values, nan=1e6, posinf=1e6, neginf=-1e6)
    
    # The goal is minimization. 
    # The candidate with the smallest sampled value is the one we want to pick.
    # The contract requires "higher meaning evaluate next".
    # So we return -sampled_values.
    score = -sampled_values
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score

# EVOLVE-BLOCK-END
