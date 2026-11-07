"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw a single sample from the GP posterior
    # and select the candidate that minimizes (sample / cost).
    # Since we want to return scores where higher = evaluate next,
    # we return - (sample / cost).
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw independent standard normal samples for each candidate.
    # Given the small length scale (0.2) and 2000 candidates,
    # assuming independence is a better approximation for Thompson Sampling
    # than a single global shift.
    z = np.random.standard_normal(size=mu.shape[0])
    
    # GP posterior sample: sample = mu + sigma * z
    # We want to minimize the sampled value relative to the cost.
    sample = mu + sigma * z
    
    # Compute score: higher is better, so we negate the ratio
    # Avoid division by zero or very small costs
    cost_safe = np.maximum(cost, 1e-9)
    
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
