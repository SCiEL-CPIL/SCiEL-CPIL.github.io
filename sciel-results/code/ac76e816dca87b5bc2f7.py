"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix for ant colony optimization of capacitated vehicle routing.
    The heuristic should be larger for more attractive edges.
    """
    # Use inverse distance as the primary heuristic, which is standard for ACO.
    # We add a small epsilon to avoid division by zero.
    heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Ensure the heuristic is finite and positive
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
