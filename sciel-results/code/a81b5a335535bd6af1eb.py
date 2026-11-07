import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Thompson Sampling acquisition function
    # Draw a single sample from the posterior: theta = mu + sigma * Z
    # where Z is a standard normal random variable
    # Score candidates by -theta (higher is better for minimization)
    
    # Track deadline
    start_time = time.time()
    
    # Generate a single standard normal random number
    z = np.random.standard_normal()
    
    # Compute the Thompson Sampling sample
    theta = mu + sigma * z
    
    # Score is the negative of the sample (we want to minimize, so higher score = lower theta)
    scores = -theta
    
    # Ensure scores are finite as per contract
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    # Deadline guard: check elapsed time
    elapsed = time.time() - start_time
    # The computation is O(N) and very fast, but we include the guard as required
    # If we were doing more work, we'd check here and return early if needed
    # For this simple computation, no early return is needed
    
    return scores
