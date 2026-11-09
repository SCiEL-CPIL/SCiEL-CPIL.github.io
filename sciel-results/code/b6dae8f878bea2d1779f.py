"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Pareto-frontier guided exploration:
    # 1. Find the candidate with lowest predicted mean (exploitation anchor)
    # 2. Score = sigma * normalized_distance_from_anchor
    # This prioritizes high-uncertainty regions far from the best exploitation point
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    # Find the index of the candidate with the lowest predicted mean
    anchor_idx = np.argmin(mu)
    anchor_mu = mu[anchor_idx]
    
    # Compute distance from anchor in prediction space
    # Use mu as the coordinate for distance (normalized)
    mu_range = np.ptp(mu)
    if mu_range > 1e-10:
        normalized_dist = np.abs(mu - anchor_mu) / mu_range
    else:
        normalized_dist = np.ones_like(mu)
    
    # Score: product of uncertainty and distance from exploitation anchor
    # This encourages exploration of uncertain regions that are far from the best mean
    score = sigma * (0.5 + normalized_dist)
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
