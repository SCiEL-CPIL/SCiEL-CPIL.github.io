"""Optimized heuristics for ACO CVRP with instance-normalized inverse distance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy:
    - Base: Inverse distance (1/d), which is the strongest baseline.
    - Refinement: Instance-wise normalization.
      The pheromone starts at 1.0. If the heuristic values are very large 
      (dense instances) or very small (sparse instances), the relative 
      influence of pheromone vs heuristic is skewed. 
      Normalizing the heuristic matrix so that the mean off-diagonal value 
      is approximately 1.0 ensures a balanced starting point for all instances,
      allowing the pheromone update mechanism to work consistently.
    """
    # Safety margin: The evaluator kills after 330s. 
    # This function is called once per instance. It should be fast (< 1ms).
    # The deadline guard is primarily for safety in case of unexpected input sizes.
    deadline = time.time() + 290.0
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n == 0:
        return np.array([])
        
    if d.shape != (n, n):
        # Fallback to safe default if shape is wrong, though contract says same shape
        return 1.0 / (d + 1e-9)
        
    eps = 1e-9
    
    # Base heuristic: Inverse distance with exponent 1.5
    # Using 1/d^1.5 makes the heuristic more discriminative,
    # pushing ants more strongly toward closer nodes
    eta = 1.0 / ((d + eps) ** 1.5)
    
    # Normalize the heuristic matrix
    # We want the mean of the off-diagonal elements to be 1.0
    # This balances the initial pheromone (1.0) with the heuristic
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])
    
    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        # Fallback if normalization fails
        eta = 1.0 / ((d + eps) ** 1.5)
        
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
