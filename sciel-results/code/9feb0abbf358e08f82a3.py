# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson sampling based acquisition: sample from the posterior,
    then choose the candidate maximizing sampled value per unit cost.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior: theta = mu + sigma * z, where z ~ N(0,1)
    # This is a single Thompson sampling draw
    z = np.random.standard_normal(mu.shape)
    sampled_values = mu + sigma_safe * z
    
    # We want to minimize the function, so we want low sampled values.
    # The acquisition score should be high for candidates we want to evaluate.
    # So we take negative of the sampled value (since lower is better for the objective).
    # Then divide by cost to get cost-aware score.
    
    # To handle cost properly: score = -sampled_value / cost
    # But we must ensure finite outputs.
    
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -sampled_values / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
