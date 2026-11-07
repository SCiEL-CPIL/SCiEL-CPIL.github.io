"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Use the maximum weight across constraints as the cost metric
    # This accounts for the most restrictive constraint for each item
    max_weights = np.max(weights, axis=0)
    
    # Heuristic value: Prize divided by the bottleneck weight
    # This encourages selection of items with high value-to-bottleneck-weight ratio
    eta = prizes / (max_weights + 1e-9)
    
    # Ensure all values are positive and finite to avoid ACO issues
    eta = np.where(eta > 0, eta, 1e-9)
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
