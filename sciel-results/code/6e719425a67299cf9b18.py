"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    This heuristic combines:
    1. Inverse distance (standard)
    2. Angle-based preference for consistent routing direction
    3. Demand-aware adjustment to prefer customers that fit well
    
    Returns a matrix where larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    
    # Base: inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Angle-based heuristic: prefer edges that maintain directional consistency
    # For each pair (i, j), compute the angle between the vector from depot to i
    # and the vector from i to j. Prefer small angles (smooth turns).
    # Actually, a better angle heuristic: prefer edges where the direction from i to j
    # is consistent with the "flow" around the depot.
    
    # Compute angles of all points relative to depot
    depot_coords = coordinates[0]
    angles = np.zeros(n)
    for i in range(1, n):
        dx = coordinates[i, 0] - depot_coords[0]
        dy = coordinates[i, 1] - depot_coords[1]
        angles[i] = np.arctan2(dy, dx)
    
    # For edge (i, j), compute the angular difference
    # Prefer edges where the angle changes smoothly (small |angle_j - angle_i|)
    # This encourages tour-like structures
    
    angle_diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
    # Wrap around 2*pi
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Angular attractiveness: prefer small angle differences
    angle_attr = 1.0 / (angle_diff + 0.1)
    
    # Demand-aware heuristic: prefer going to customers with moderate demand
    # when coming from depot (to balance loads), but this is complex.
    # Instead, use a simple demand-based weight: slightly prefer lower demand
    # customers to leave room for more stops.
    demand_factor = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Slight preference for lower demand destinations
                demand_factor[i, j] = 1.0 / (1.0 + demands[j] * 0.05)
    
    # Combine heuristics
    # Weight the components
    heuristic = inv_dist * 0.7 + angle_attr * 0.2 + demand_factor * 0.1
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
