"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Structural change from deterministic EI/UCB to stochastic sampling.
    """
    t_start = time.time()
    
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling:
    # Sample from the posterior distribution N(mu, sigma^2)
    y_samples = np.random.normal(mu, sigma_safe)
    
    # Cost-awareness:
    # We want to minimize the function value plus a cost penalty.
    # Score = - (y_sample + lambda * cost)
    # Higher score is better.
    
    # Calibrate lambda.
    # Cost range is [1, 2].
    # Function values are around -0.8 to ~1.
    # A lambda of 0.5 makes 1 unit of cost equivalent to 0.5 units of function value.
    # This encourages finding low-cost points unless the predicted function value
    # is significantly better.
    lam = 0.5
    
    # Dynamic lambda adjustment based on f_best?
    # If f_best is very good (close to -0.8), we should be more conservative (higher lam).
    # If f_best is bad, we should be more exploratory (lower lam).
    # f_min = -0.8
    # regret = f_best - (-0.8)
    # If regret is high, we need to find a better point.
    # Let's keep lam fixed for simplicity in this structural change.
    
    scores = -(y_samples + lam * cost)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
