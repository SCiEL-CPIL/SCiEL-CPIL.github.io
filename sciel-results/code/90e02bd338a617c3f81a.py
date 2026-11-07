"""Improved heuristics for ACO TSP."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for the given distance matrix.
    
    Uses a combination of inverse distance (with regularization) and a 
    smoothing/normalization to provide strong attraction to short edges
    while avoiding numerical issues.
    
    Args:
        distance_matrix: (n, n) array of pairwise distances
        
    Returns:
        (n, n) array of heuristic values (larger = more attractive)
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.copy()
    
    # Use inverse squared distance with small epsilon
    # This sharpens the preference for short edges
    eps = 1e-9
    eta = 1.0 / (d + eps)**2
    
    # Set diagonal to 0 (will become 1e-9)
    np.fill_diagonal(eta, 0.0)
    
    # Ensure non-finite or non-positive values are 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta

# EVOLVE-BLOCK-END
