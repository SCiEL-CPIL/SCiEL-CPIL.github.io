"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Depot-pressure heuristic: prioritizes edges that cluster high-demand nodes
    near the depot and scales by demand efficiency.
    
    Design:
    - Base: inverse distance (closer is better)
    - Demand pressure: nodes with higher demand get a boost when visited early
      (when truck is full), modeled as demand[i] * demand[j] / capacity
    - Depot proximity bonus: nodes closer to depot get slight boost since they
      are easier to serve in first trip
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Inverse distance base
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Demand efficiency: product of demands normalized by capacity
    # This creates pressure to visit high-demand nodes together
    demand_prod = np.outer(demands, demands) / capacity
    
    # Depot proximity: distance from each node to depot (node 0)
    depot_dist = distance_matrix[0, :]  # shape (n,)
    depot_proximity = 1.0 / (depot_dist + 1e-9)
    
    # Combine: 
    # - For depot->customer edges (row 0): boost by demand and proximity
    # - For customer->customer edges: boost by demand product (clustering)
    # - For customer->depot edges (col 0): slight boost for proximity
    
    heuristic = inv_dist.copy()
    
    # Add demand clustering bonus for customer-to-customer edges
    # Scale by 0.5 to not overwhelm distance
    customer_mask = np.ones((n, n), dtype=bool)
    customer_mask[0, :] = False  # exclude depot row
    customer_mask[:, 0] = False  # exclude depot column
    
    heuristic[customer_mask] += 0.5 * demand_prod[customer_mask] * inv_dist[customer_mask]
    
    # Depot to customer: add demand-based priority (high demand = visit early)
    heuristic[0, 1:] += 0.3 * demands[1:] * inv_dist[0, 1:]
    
    # Customer to depot: slight proximity boost
    heuristic[1:, 0] += 0.2 * depot_proximity[1:] * inv_dist[1:, 0]
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
