"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    # Ensure inputs are numpy arrays
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    
    # Start with inverse distance as a base heuristic
    # Add small epsilon to avoid division by zero
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Add a demand-aware component: edges to customers with lower demand
    # are slightly more attractive to help with capacity constraints
    # This is a heuristic bias that helps ants make better choices
    
    # Compute a demand penalty factor for each customer
    # Lower demand customers get a slight boost
    demand_factor = 1.0 + 0.1 * (1.0 - demands / max(demands.max(), 1e-9))
    demand_factor[0] = 1.0  # Depot has no demand consideration
    
    # Apply demand factor to columns (destination customers)
    heuristic = base_heuristic * demand_factor[np.newaxis, :]
    
    # Add a coordinate-based component: edges that go in a more "direct" 
    # direction relative to the depot might be slightly favored
    # This is a subtle geometric heuristic
    
    # Compute distance from depot for each node
    depot_coords = coordinates[0]
    dist_from_depot = np.sqrt(np.sum((coordinates - depot_coords) ** 2, axis=1))
    dist_from_depot[0] = 0.0
    
    # Edges that don't move too far away from depot are slightly favored
    # This encourages more compact routes
    max_dist = max(dist_from_depot.max(), 1e-9)
    proximity_factor = 1.0 + 0.05 * (1.0 - dist_from_depot / max_dist)
    proximity_factor[0] = 1.0
    
    heuristic = heuristic * proximity_factor[np.newaxis, :]
    
    # Ensure all values are positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
