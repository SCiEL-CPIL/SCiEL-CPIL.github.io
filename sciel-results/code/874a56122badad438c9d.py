"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Angle-consistency + distance + demand-aware heuristic for CVRP ACO.
    
    Structural change: Instead of pure inverse-distance, we use a composite heuristic
    that rewards:
    1. Directional consistency (angle between (i->j) and (depot->i) or (depot->j))
    2. Short distance
    3. Lower demand (to allow more stops per route)
    
    This models route segment coherence rather than just proximity.
    """
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates)
    demands = np.asarray(demands)
    
    # Precompute vectors from depot to each node
    depot = coords[0]
    vectors = coords - depot  # shape (n, 2)
    
    # Compute angles between vectors for directional consistency
    # angle_consistency[i,j] = how aligned the direction i->j is with continuing the route
    # We use the dot product of (coords[i] - depot) and (coords[j] - depot) normalized
    # This favors going to customers in similar angular sectors
    
    # Normalize vectors (avoid division by zero)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    unit_vectors = vectors / norms
    
    # Cosine similarity matrix: favors customers in similar angular direction from depot
    cos_sim = unit_vectors @ unit_vectors.T  # shape (n, n)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Map cosine similarity to [0.1, 1.0] range - higher means more aligned
    angle_score = 0.1 + 0.9 * (cos_sim + 1.0) / 2.0
    
    # Distance component: inverse distance, normalized
    dist_comp = 1.0 / (distance_matrix + 1e-9)
    # Normalize distance component to [0, 1] range
    dist_max = dist_comp.max(axis=1, keepdims=True)
    dist_max = np.maximum(dist_max, 1e-9)
    dist_score = dist_comp / dist_max
    
    # Demand component: lower demand is slightly better (more flexibility)
    # Map demand [1,9] to [0.8, 1.0] - lower demand gets higher score
    demand_score = 1.0 - 0.02 * (demands[None, :] - 1.0)  # shape (1, n)
    demand_score = np.maximum(demand_score, 0.5)
    
    # Combine: weighted product for discrimination
    # Power exponents control the relative importance and discrimination
    heuristic = (angle_score ** 1.5) * (dist_score ** 2.0) * (demand_score ** 0.5)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
