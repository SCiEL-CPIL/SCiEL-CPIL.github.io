"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    Uses a combination of distance-based heuristic and demand-aware weighting.
    """
    # Ensure we have the right shape and convert to float
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Start with inverse distance as base heuristic
    # Add small epsilon to avoid division by zero
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Create a demand-aware adjustment:
    # For each edge (i, j), if j is a customer with high demand, it should be less attractive
    # especially if we're close to capacity. However, since the heuristic is computed once
    # and doesn't know the current load, we use a static adjustment based on demand.
    
    # Normalize demands: higher demand -> less attractive
    # Use a soft penalty: 1 / (1 + alpha * demand_j)
    # where alpha is a tuning parameter
    alpha = 0.1
    demand_penalty = 1.0 / (1.0 + alpha * dem)  # shape (n,)
    
    # Apply the penalty to each column (destination node)
    # heuristic[i, j] *= demand_penalty[j]
    # But we need to be careful: the depot (node 0) has demand 0, so penalty is 1.
    
    adjusted_heuristic = base_heuristic * demand_penalty[np.newaxis, :]
    
    # Ensure all values are positive and finite
    adjusted_heuristic[~np.isfinite(adjusted_heuristic)] = 1e-9
    adjusted_heuristic[adjusted_heuristic <= 0] = 1e-9
    
    return adjusted_heuristic
# EVOLVE-BLOCK-END
