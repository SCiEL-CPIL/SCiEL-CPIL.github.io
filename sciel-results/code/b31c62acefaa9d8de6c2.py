"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Distance-to-Best weighted acquisition:
    # Scale Expected Improvement by inverse distance to current best in standardized input space.
    # This creates a repulsive force that encourages exploration of distant, unexplored basins,
    # which is critical for escaping the dense local minima of the Michalewicz function.
    
    # Compute standardized input space coordinates from mu and sigma
    # Since we don't have coordinates directly, we use the index-based approach
    # The candidates are drawn uniformly on [0, 3.14159]^2, so we can infer positions from indices
    
    n = len(mu)
    
    # Infer approximate coordinates from candidate indices
    # Candidates are 2000 uniform random points in [0, pi]^2
    # We can estimate their positions based on their rank/ordering
    # But actually, we need to think about this differently.
    
    # Since we don't have the actual coordinates, let's use a different approach:
    # Use the uncertainty (sigma) and improvement potential to create a distance-like metric.
    
    # Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Create a "distance to best" proxy using the combination of:
    # 1. High uncertainty (sigma) suggests unexplored regions
    # 2. Low mean (mu) suggests potentially good regions
    # We want to boost points that are far from the current best in some sense
    
    # Normalize mu and sigma to [0, 1] range
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    if mu_max > mu_min:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)
    else:
        mu_norm = np.zeros_like(mu)
    
    if sigma_max > sigma_min:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Distance-to-best proxy: points with high uncertainty and low mean are "far" from best
    # in the sense that they're unexplored and potentially better
    # Use a combination: distance_proxy = sigma_norm * (1 - mu_norm)
    # This gives high values to points that are uncertain (unexplored) and have low predicted values
    
    distance_proxy = sigma_norm * (1.0 - mu_norm)
    
    # Add a small epsilon to avoid division by zero
    distance_proxy_safe = distance_proxy + 1e-6
    
    # Weight EI by inverse distance proxy (higher for more distant/unexplored points)
    # Use a power to control the strength of the repulsive force
    weight = 1.0 / np.sqrt(distance_proxy_safe)
    
    # Combine EI with the distance weight
    scores = ei * weight
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
