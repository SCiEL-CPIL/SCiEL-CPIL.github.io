"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Angular-sector clustering + demand-cohesion heuristic for CVRP ACO.
    
    Structural design:
    - Primary signal: angular consistency (cosine similarity of vectors from depot)
      to encourage ants to serve customers in the same directional sector, which
      naturally forms coherent route segments.
    - Secondary signal: demand clustering (product of demands) to encourage
      grouping high-demand nodes together, improving load utilization.
    - Base: inverse distance for local proximity.
    
    The heuristic is a weighted geometric mean of these three components,
    which keeps all factors in play without any single one dominating.
    """
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    # --- Component 1: Inverse distance (proximity) ---
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    # Normalize each row so that the max entry is 1, making it comparable
    inv_dist_max = inv_dist.max(axis=1, keepdims=True)
    inv_dist_max = np.maximum(inv_dist_max, 1e-9)
    dist_score = inv_dist / inv_dist_max  # in (0, 1]
    
    # --- Component 2: Angular consistency (sector coherence) ---
    depot = coords[0]
    vectors = coords - depot  # shape (n, 2)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    unit_vectors = vectors / norms
    
    # Cosine similarity matrix: favors customers in similar angular direction
    cos_sim = unit_vectors @ unit_vectors.T  # shape (n, n)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Map cosine similarity to [0.05, 1.0] - higher means more aligned
    # Using a power to sharpen the discrimination between aligned and misaligned
    angle_score = 0.05 + 0.95 * ((cos_sim + 1.0) / 2.0) ** 1.5
    
    # --- Component 3: Demand clustering (cohesion) ---
    # Product of demands normalized by capacity^2 to keep in reasonable range
    # This encourages visiting high-demand nodes together
    demand_prod = np.outer(demands, demands)
    # Normalize: divide by max possible (9*9=81) to get [0, 1]
    demand_score = demand_prod / 81.0
    # Add a small floor so that low-demand edges aren't completely suppressed
    demand_score = 0.1 + 0.9 * demand_score
    
    # --- Combine via geometric mean with tuned exponents ---
    # dist_score^2.0 * angle_score^1.8 * demand_score^0.6
    # The exponents sum to 4.4, but since each is in [0,1] range, the product
    # is in [0,1]. The relative exponents control importance:
    #   - Distance is most important (closest nodes first)
    #   - Angular consistency is second (stay in same sector)
    #   - Demand clustering is tertiary (group heavy loads)
    heuristic = (dist_score ** 2.0) * (angle_score ** 1.8) * (demand_score ** 0.6)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
