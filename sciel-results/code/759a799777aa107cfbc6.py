"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    
    Args:
        distance_matrix: NxN matrix of distances.
        coordinates: Nx2 array of coordinates.
        demands: Array of demands for each node (0 is depot, demand 0).
        capacity: Vehicle capacity.
        
    Returns:
        NxN matrix of heuristic values (edge attractiveness).
    """
    # Ensure inputs are numpy arrays
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # Add a small epsilon to avoid division by zero for diagonal elements
    # and very small distances
    eps = 1e-9
    d = distance_matrix + eps
    
    # Parameters for the heuristic
    # alpha: distance exponent. Higher alpha makes ants prefer shorter distances more strongly.
    # beta: demand exponent. Higher beta makes ants prefer customers with higher demand.
    alpha = 2.0
    beta = 1.0
    
    # Calculate base distance heuristic: 1 / d^alpha
    # Shape: (N, N)
    dist_heur = 1.0 / (d ** alpha)
    
    # Calculate demand heuristic: demand^beta
    # Shape: (N,)
    # We want eta[i, j] to depend on demand[j] (the destination)
    demand_heur = demands ** beta
    
    # Combine: eta[i, j] = dist_heur[i, j] * demand_heur[j]
    # Broadcast demand_heur along the rows
    heuristic_matrix = dist_heur * demand_heur[np.newaxis, :]
    
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # Since d > 0 and demands >= 0, the values are non-negative.
    # If demand is 0 (depot), the heuristic is 0.
    # We replace non-positive values with 1e-9 to ensure they are positive as required.
    heuristic_matrix[heuristic_matrix <= 0] = 1e-9
    
    # Ensure the matrix has the correct shape and is finite
    if heuristic_matrix.shape != distance_matrix.shape:
        raise ValueError("Heuristic matrix shape mismatch")
        
    if not np.all(np.isfinite(heuristic_matrix)):
        # Replace non-finite values with 1e-9
        heuristic_matrix[~np.isfinite(heuristic_matrix)] = 1e-9
        
    return heuristic_matrix
# EVOLVE-BLOCK-END
