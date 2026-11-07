"""Rewritten program using Thompson Sampling for cost-aware BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Uses a reduced exploration rate (gamma=0.5) to balance exploration
    and exploitation, preventing excessive sampling of high-uncertainty
    regions in the early stages of optimization.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division/sampling issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Exploration parameter: reduce variance of the sample to favor exploitation
    gamma = 0.5
    
    # Sample theta_i from the GP posterior for each candidate
    # theta_i = mu_i + gamma * sigma_i * z_i, where z_i ~ N(0, 1)
    z = np.random.standard_normal(size=mu.shape)
    theta = mu + gamma * sigma * z
    
    # Calculate cost-normalized scores
    # We want to minimize theta (function value) and cost.
    # Score = -theta / cost.
    scores = -theta / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
