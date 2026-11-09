"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO.
    Combines a sharpened inverse-distance heuristic with a demand-aware bias
    to encourage visiting high-demand nodes earlier, improving packing efficiency.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # Base heuristic: sharpened inverse distance (d^-3)
    # This strongly biases towards nearest-neighbor moves, reducing long jumps.
    base_heuristic = 1.0 / safe_dist ** 3
    
    # Demand-aware component:
    # Encourage visiting high-demand nodes earlier in the route.
    # This helps with capacity constraints by ensuring large items are placed
    # when the vehicle is most empty, potentially allowing better packing of
    # remaining smaller items.
    # weights[j] = 1 + demand[j]
    # Using (1 + demand) ensures weight > 1 and scales linearly with demand.
    weights = 1.0 + demands  # Shape (n,)
    
    # Broadcasting: weights[None, :] / safe_dist is not quite right for the base.
    # We want to multiply the base heuristic by a demand factor.
    # If we multiply eta[i,j] by weight[j], we are saying "edges to high-demand nodes are more attractive".
    # Let's try: eta[i,j] = base_heuristic[i,j] * weights[j]
    
    # However, we need to be careful. The current program (d^-3) scored 1.476.
    # Inspiration 1 (d^-1 * demand) scored 1.0132.
    # The d^-3 is much stronger. Adding a demand multiplier might be too strong or too weak.
    # Let's use a softer demand bias.
    # demand_factor[j] = 1 + alpha * (demand[j] / max_demand)
    # Or simply 1 + demand[j] * small_constant.
    
    # Let's try: eta[i,j] = base_heuristic[i,j] * (1 + 0.1 * demands[j])
    # This keeps the strong distance bias but adds a slight preference for high-demand nodes.
    
    max_demand = np.max(demands)
    if max_demand > 0:
        demand_norm = demands / max_demand
    else:
        demand_norm = demands
        
    # Soft bias: 1 + 0.2 * demand_norm
    # This ranges from 1.0 (zero demand) to 1.2 (max demand).
    demand_factor = 1.0 + 0.2 * demand_norm
    demand_factor[0] = 1.0  # Depot has no demand
    
    heuristic = base_heuristic * demand_factor[np.newaxis, :]
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
