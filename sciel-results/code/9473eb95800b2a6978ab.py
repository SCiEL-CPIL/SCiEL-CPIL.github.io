import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Explicit deadline guard
    start_time = time.time()
    # We don't know the exact global elapsed time, but we can ensure this function
    # doesn't hang. Given the vectorization, it should be instantaneous.
    
    # UCB (Upper Confidence Bound) for minimization:
    # We want to maximize the score.
    # score = mu + beta * sigma
    # A larger beta encourages exploration.
    # With only ~25 optimization steps, a moderate beta is appropriate.
    # Beta = 1.0 is a standard starting point.
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute UCB scores
    scores = mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check if we are close to the deadline (though unlikely for this op)
    # If this function were complex, we would check here.
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        # Fallback to exploitation if computation is unexpectedly slow
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
    return scores
