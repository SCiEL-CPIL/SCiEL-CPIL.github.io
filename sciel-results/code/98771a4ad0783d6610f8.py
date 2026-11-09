"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO.
    Uses a sharpened inverse-distance heuristic with a softer demand-aware bias.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # Base heuristic: sharpened inverse distance (d^-3)
    base_heuristic = 1.0 / safe_dist ** 3
    
    # Demand-aware component:
    # Encourage visiting high-demand nodes earlier in the route.
    max_demand = np.max(demands)
    if max_demand > 0:
        demand_norm = demands / max_demand
    else:
        demand_norm = demands
        
    # Softer bias: 1 + 0.1 * demand_norm
    # This ranges from 1.0 (zero demand) to 1.1 (max demand).
    demand_factor = 1.0 + 0.1 * demand_norm
    demand_factor[0] = 1.0  # Depot has no demand
    
    heuristic = base_heuristic * demand_factor[np.newaxis, :]
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
