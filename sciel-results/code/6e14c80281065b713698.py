import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    The Weierstrass function is highly multimodal. With a limited budget (30 evaluations),
    standard Expected Improvement (EI) tends to get stuck in local minima.
    Aggressive exploration via UCB allows the algorithm to jump between basins of attraction.
    
    Based on observed performance improvements, a beta value of 2.5 is effective.
    
    Args:
        mu: array of shape (2000,) containing the GP mean predictions.
        sigma: array of shape (2000,) containing the GP standard deviation predictions.
        f_best: float, the lowest observed function value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    # though with noise=1e-6 it should be fine.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB for minimization:
    # We want to find points with low mu (likely good) and high sigma (uncertain).
    # Score = -mu + beta * sigma
    # We maximize this score.
    
    # The inspiration suggests beta=2.5 is significantly better than baseline EI (score 1.0 vs 2.155).
    beta = 2.5
    
    # Calculate scores
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite. If any are not, replace with a very low value
    # so they are never selected.
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
    # Time guard: If we are running out of time, return immediately.
    # This is extremely fast, but good practice per instructions.
    if time.time() - start_time > 290:
         return scores

    return scores
