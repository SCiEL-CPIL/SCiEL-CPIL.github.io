"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Strategy: Combine inverse Euclidean distance with a penalty for long-range edges
    and a bonus for short-range edges, creating a strong local bias that helps ants
    build tours without long jumps.
    
    The heuristic is:
      h[i,j] = 1 / (d[i,j]^alpha) * (1 + beta * exp(-gamma * d[i,j]))
    
    where alpha controls distance sensitivity, beta and gamma control the local bonus.
    """
    n = distance_matrix.shape[0]
    
    # Work with a copy to avoid modifying input
    d = distance_matrix.copy()
    
    # Handle zero diagonal (self-distance) - set to 0, will be masked later
    np.fill_diagonal(d, 0.0)
    
    # Small epsilon to avoid division by zero
    eps = 1e-12
    d_safe = np.maximum(d, eps)
    
    # Parameters tuned for 50 points in unit square
    alpha = 2.0       # Distance exponent (stronger penalty for far edges)
    beta = 1.5        # Local bonus strength
    gamma = 8.0       # Local bonus decay rate
    
    # Base inverse distance term
    base = 1.0 / (d_safe ** alpha)
    
    # Local bonus: rewards very short edges
    local_bonus = 1.0 + beta * np.exp(-gamma * d_safe)
    
    # Combine
    h = base * local_bonus
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(h, 0.0)
    
    # Ensure all off-diagonal entries are positive and finite
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Normalize so the maximum is 1 (helps with numerical stability in ACO)
    max_h = np.max(h)
    if max_h > 0 and np.isfinite(max_h):
        h = h / max_h
    
    # Final safety: ensure positive, finite, correct shape
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
