"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    The UCB acquisition function balances exploitation (low mean) and exploration
    (high standard deviation). For minimization, we want to find points with low
    predicted values, but also points with high uncertainty to cover the search space.
    
    Score = -mu + kappa * sigma
    
    A higher score indicates a more promising candidate.
    """
    start_time = time.time()
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter. 
    # Based on inspiration, kappa=1.0 performed well.
    # We can tune this, but 1.0 is a standard starting point.
    kappa = 1.0
    
    # Compute UCB score for minimization
    # We want to minimize f, so we prefer low mu.
    # The term -mu rewards low mu.
    # The term + kappa * sigma rewards high uncertainty.
    score = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 295.0:
        return score
        
    return score
# EVOLVE-BLOCK-END
