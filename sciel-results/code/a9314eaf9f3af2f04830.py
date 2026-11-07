"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with cost normalization.
    
    Samples from the posterior GP and selects the candidate that minimizes
    the sampled objective value per unit cost, naturally balancing exploration
    and exploitation.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from the posterior
    # The GP posterior is mu + sigma * z where z ~ N(0, 1)
    # We sample z from standard normal
    z_sample = np.random.standard_normal(len(mu))
    
    # Sampled objective values
    sampled_values = mu + sigma_safe * z_sample
    
    # Cost-aware Thompson Sampling: minimize sampled_value / cost
    # Higher score = better (we want to maximize the acquisition score)
    # So we negate: score = -sampled_value / cost
    ts_score = -sampled_values / cost_safe
    
    # Add a small exploitation bias: prefer points with lower mu
    # This helps when Thompson Sampling might be too exploratory
    # Use a small weight to not dominate the TS signal
    mu_component = -mu / cost_safe
    
    # Combine: primarily TS with small exploitation bias
    # Weight 0.85 for TS, 0.15 for exploitation
    score = 0.85 * ts_score + 0.15 * mu_component
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    if time.time() - t0 > 280:
        # Return simple cost-normalized negated mu as fallback
        fallback = -mu / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
