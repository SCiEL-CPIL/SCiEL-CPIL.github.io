"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track evaluation count and start time for deadline guard
_eval_count = 0
_start_time = None

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
    # Thompson Sampling on 2000 points is O(N) if we just sample.
    # The GP posterior is already computed by the caller (mu, sigma).
    # So we just need to sample from N(mu, sigma^2).
    
    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle NaNs and Infs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Check time remaining. 
    # If we have spent too much time, we might be in trouble.
    # But since we are just sampling, it's very fast.
    # We will just proceed.
    
    # Thompson Sampling:
    # Draw a single sample from the posterior distribution at each candidate point.
    # y_i ~ N(mu_i, sigma_i^2)
    # We want to minimize f, so we want to select the point with the lowest sample value.
    # Therefore, the "score" to maximize is -y_i.
    
    # To make it slightly more robust against pure noise in early stages,
    # we can mix in a small amount of exploration or just rely on the sampling.
    # Standard TS is known to work well for BO.
    
    # Generate random normal variables
    # Use a fixed seed? No, the evaluator runs multiple seeds. 
    # The global state should probably not use a fixed seed to allow diversity, 
    # but for reproducibility within a run, we rely on the global RNG state.
    # However, to be safe and deterministic relative to the input if possible,
    # we could use a seed based on _eval_count? 
    # No, standard practice is to use the global RNG.
    
    # Draw samples
    samples = mu + sigma * np.random.randn(mu.shape[0])
    
    # The acquisition score is the negative of the sample, because we want to MAXIMIZE the score
    # to find the MINIMUM of the function.
    scores = -samples
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return scores

# EVOLVE-BLOCK-END
