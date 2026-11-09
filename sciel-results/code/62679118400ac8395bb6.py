"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Uses a structural approach:
    - Strong inverse distance bias (exponent 3) to encourage short edges
    - Angular proximity bonus: prefer edges between nodes in similar angular
      sectors from the depot (reduces route crossings)
    - Demand-aware scaling: slightly boost edges to lower-demand nodes to
      allow more customers per route
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    
    # Strong inverse distance: exponent 3 gives strong nearest-neighbor bias
    base = 1.0 / (distance_matrix + 1e-9) ** 3
    
    # Angular proximity: compute angle from depot to each customer
    depot = coordinates[0]
    # For customers (indices 1 to n-1)
    dx = coordinates[1:, 0] - depot[0]
    dy = coordinates[1:, 1] - depot[1]
    angles = np.arctan2(dy, dx)  # shape (n-1,)
    
    # Build angular similarity matrix for customer-to-customer edges
    # For edge i->j (both customers), bonus based on angular difference
    # Use cos of angle difference: 1 when same direction, -1 when opposite
    # We want a mild bonus, so use 1 + alpha * cos(da)
    alpha = 0.5
    # angles_diff[i,j] = angles[i] - angles[j] for i,j in 1..n-1
    angles_diff = angles[:, None] - angles[None, :]
    # Normalize to [-pi, pi]
    angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
    angular_bonus_cc = 1.0 + alpha * np.cos(angles_diff)  # shape (n-1, n-1)
    
    # For depot->customer and customer->depot, no angular bonus (just 1.0)
    
    # Build full angular bonus matrix
    angular_bonus = np.ones((n, n))
    angular_bonus[1:, 1:] = angular_bonus_cc
    
    # Demand-aware scaling:
    # Lower demand customers are "easier" to pack, so slightly boost edges to them
    # This encourages routes to fill with more customers
    # Normalize demand: d_norm = demand / max_demand
    max_demand = np.max(demands[1:]) if n > 1 else 1.0
    demand_norm = demands / max_demand  # shape (n,)
    
    # For edges TO customer j, multiply by (1 + beta * (1 - demand_norm[j]))
    # This boosts edges to low-demand customers
    beta = 0.3
    demand_scale = np.ones((n, n))
    for j in range(1, n):
        demand_scale[:, j] *= (1.0 + beta * (1.0 - demand_norm[j]))
    
    # Combine: base * angular_bonus * demand_scale
    result = base * angular_bonus * demand_scale
    
    # Ensure all values are positive and finite
    result = np.maximum(result, 1e-9)
    result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
