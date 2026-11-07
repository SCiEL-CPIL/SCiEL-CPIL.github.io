import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Multi-Sample Thompson Sampling heuristic.
    Samples N values from the posterior for each candidate and returns the negative
    of the minimum sample. N adapts to f_best to balance exploration and exploitation.
    """
    start_time = time.time()
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Safety check for time
    if time.time() - start_time > 290:
        # Fallback to simple LCB
        return -mu + 2.0 * sigma

    # Normalize f_best to [0, 1]
    f_max = 25.0
    f_norm = np.clip(f_best, 0.0, f_max) / f_max
    
    # Adaptive number of samples N
    # Early (f_norm ~ 1): High N (e.g., 500) -> Strong exploration (tail seeking)
    # Late (f_norm ~ 0): Low N (e.g., 1) -> Pure exploitation (mean seeking)
    N_min = 1
    N_max = 500
    N = int(N_min + (N_max - N_min) * f_norm)
    N = max(N, 1)
    
    # Generate samples: shape (num_candidates, N)
    # Using broadcasting
    samples = np.random.normal(mu[:, None], sigma[:, None], size=(mu.size, N))
    
    # Find the minimum sample for each candidate
    min_samples = np.min(samples, axis=1)
    
    # Score: higher is better, so we negate the minimum value
    scores = -min_samples
    
    # Final time check
    if time.time() - start_time > 295:
        pass # Scores are already computed, just return
        
    return scores
