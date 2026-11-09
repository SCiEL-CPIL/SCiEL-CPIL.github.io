"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Heuristic matrix for ant colony optimization of TSP.
    
    Uses a slightly softened inverse distance (d^-0.9) as the base heuristic.
    This is close to the standard 1/d but with mild softening to reduce the
    overwhelming pull of the absolute nearest neighbor, helping ants avoid
    premature convergence to small local cycles while preserving the strong
    preference for shorter edges that drives good tour construction.
    
    All values are ensured to be strictly positive and finite.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Avoid division by zero on the diagonal
    d_safe = dm.copy()
    np.fill_diagonal(d_safe, 1e9)
    
    # Slightly softened inverse distance: d^(-0.9) instead of d^(-1)
    # This keeps the strong preference for short edges while slightly
    # reducing the gap between nearest and next-nearest neighbors
    heuristic = 1.0 / (d_safe ** 0.9)
    
    # Ensure diagonal is very small (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
