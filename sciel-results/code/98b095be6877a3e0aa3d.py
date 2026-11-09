"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample a value for each candidate from N(mu, sigma^2).
    2. Compute a cost-adjusted score: sampled_value / cost.
    3. Return the negative of this score (since we want to maximize acquisition,
       and lower function values are better, so lower sampled_value/cost is better).
    
    This approach naturally handles exploration via the variance of the GP posterior
    and exploitation via the mean. The cost division penalizes expensive evaluations
    directly in the objective space, which is often more effective than penalizing
    the acquisition function itself, as it aligns the search with the actual
    cost-constrained optimization goal.
    """
    # Time guard
    t_start = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: Draw a sample from the posterior for each candidate
    # Use a fixed random seed? No, randomness is part of TS.
    # However, to ensure reproducibility in tests if needed, we could seed, 
    # but standard TS relies on the stochastic nature.
    # We generate random normal samples scaled by sigma and shifted by mu.
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Cost-adjusted objective: We want to minimize f(x)/cost(x)
    # Since the acquisition function returns scores where HIGHER is better,
    # we return the negative of the ratio.
    # Ratio = samples / cost
    # Score = - (samples / cost)
    
    scores = - (samples / cost)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Time check
    t_elapsed = time.time() - t_start
    if t_elapsed > 2.0:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
