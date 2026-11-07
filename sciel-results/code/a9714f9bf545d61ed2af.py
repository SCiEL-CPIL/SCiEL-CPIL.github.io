"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Uses a multi-criteria heuristic combining:
    1. Inverse distance (standard)
    2. Angle-based preference for consistent routing direction (sweep heuristic)
    3. Demand-aware adjustment to prefer customers that pack well
    
    Returns a matrix where larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base: inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Angle-based heuristic: prefer edges that maintain directional consistency
    # For each node i, compute the angle from depot->i to i->j
    # Small angles (smooth turns) are preferred
    depot = coordinates[0]
    
    # Compute vectors from depot to each customer
    depot_to_customer = coordinates - depot  # shape (n, 2)
    
    # For each pair (i, j), compute the angle between vector depot->i and vector i->j
    # We want to prefer edges where the direction from i to j is roughly
    # continuing in the same direction as we're going around the depot
    
    # Compute the angle between depot_to_customer[i] and (coordinates[j] - coordinates[i])
    # For efficiency, we'll compute a sweep-based preference
    
    # Sweep heuristic: for each customer i, compute the angle of vector depot->i
    # Then for edge i->j, compute the angle of vector i->j relative to the
    # tangent direction at i (perpendicular to depot->i, in the direction of increasing angle)
    
    # Compute angles of each customer relative to depot
    angles = np.arctan2(depot_to_customer[:, 1], depot_to_customer[:, 0])
    
    # For each pair (i, j), compute the angle difference between:
    # - The direction from depot to i
    # - The direction from i to j
    # We want small positive angle differences (counter-clockwise sweep)
    
    # Vector from i to j
    ij_vectors = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # (n, n, 2)
    
    # Direction from depot to i
    di_vectors = depot_to_customer[:, np.newaxis, :]  # (n, 1, 2)
    
    # Compute cosine of angle between di and ij
    di_norm = np.linalg.norm(di_vectors, axis=2)  # (n, 1)
    ij_norm = np.linalg.norm(ij_vectors, axis=2)  # (n, n)
    
    # Avoid division by zero
    di_norm_safe = np.maximum(di_norm, 1e-9)
    ij_norm_safe = np.maximum(ij_norm, 1e-9)
    
    # Cosine of angle
    cos_angle = np.sum(di_vectors * ij_vectors, axis=2) / (di_norm_safe * ij_norm_safe)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Also compute sine to determine direction (cross product in 2D)
    cross = di_vectors[:, :, 0] * ij_vectors[:, :, 1] - di_vectors[:, :, 1] * ij_vectors[:, :, 0]
    sin_angle = cross / (di_norm_safe * ij_norm_safe)
    
    # Angle in [0, 2*pi)
    angle = np.arctan2(sin_angle, cos_angle)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    
    # Prefer small angles (smooth turns in the sweep direction)
    # Use a Gaussian-like decay
    angle_pref = np.exp(-angle / 0.5)
    
    # Demand-aware: prefer customers with moderate demand that fit well
    # Higher demand customers should be visited earlier (when capacity is full)
    # This is a proxy: we don't know the ant's state, so we bias toward
    # customers whose demand is "typical" (not too small, not too large)
    # Actually, let's use a simpler approach: slightly penalize very high demand
    # nodes to encourage packing multiple smaller ones, but the ACO pheromone
    # will handle the actual routing
    
    # Demand factor: slightly favor nodes with demand closer to capacity/2
    # This encourages balanced routes
    demand_factor = 1.0 / (1.0 + np.abs(demands[np.newaxis, :] - capacity / 2) / capacity)
    
    # Combine: base * angle_pref * demand_factor
    # Normalize to keep values reasonable
    heuristic = inv_dist * angle_pref * demand_factor
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize to have mean around 1
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
    
    return heuristic
# EVOLVE-BLOCK-END
