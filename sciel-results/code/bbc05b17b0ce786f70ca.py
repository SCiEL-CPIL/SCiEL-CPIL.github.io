"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Structural change: Replaces deterministic LCB/EI hybrids with stochastic Thompson Sampling.
    This approach samples from the posterior predictive distribution for each candidate
    and selects the one with the lowest sampled value, adjusted for cost.
    
    - Sample: theta ~ N(mu, sigma^2)
    - Score: -theta / cost
    - Maximizing score corresponds to minimizing sampled value per unit cost.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
        
    elapsed = time.time() - acquisition._start_time
    if elapsed > 290:
        # Fallback to simple LCB or constant if time is tight
        # Just return a cheap score to satisfy the interface
        return 1.0 / cost

    # Clip sigma to avoid numerical issues in normal sampling
    # sigma is standard deviation, should be >= 0
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Draw one sample per candidate
    # We use the standard normal distribution scaled by sigma
    # rng = np.random.default_rng() # Using global seed might be safer or just default
    # However, using np.random.normal is standard and fast
    samples = np.random.normal(mu, sigma_safe)
    
    # Cost-aware score
    # We want to minimize the function value.
    # Cost is higher for points further from the center (scaled to [0,1]).
    # The minimum of Griewank is at the center (0 in original, 0.5 in scaled).
    # So low cost is a good prior.
    # Score = -sample / cost
    # This penalizes high cost and high sampled values.
    
    # Ensure cost is not zero (it's >= 1 by definition, but safe check)
    cost_safe = np.maximum(cost, 1e-8)
    
    score = -samples / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
# EVOLVE-BLOCK-END
