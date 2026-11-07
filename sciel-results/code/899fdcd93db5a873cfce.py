"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw independent samples from the GP posterior
    # for each candidate to properly account for local uncertainty.
    # We minimize the sample divided by cost.
    # Score is higher for candidates with lower expected cost-adjusted value.
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw independent standard normal samples for each candidate
    z = np.random.standard_normal(size=len(mu))
    
    # GP posterior sample: sample = mu + sigma * z
    sample = mu + sigma * z
    
    # Compute score: we want to minimize (sample / cost), so we negate it
    # to get a "higher is better" score.
    cost_safe = np.maximum(cost, 1e-9)
    
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
