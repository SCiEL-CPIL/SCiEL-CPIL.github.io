"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO of CVRP.
    
    Combines:
    1. Strong inverse-distance base (d^-3) for nearest-neighbor bias.
    2. Clarke-Wright savings-based boost: edges that appear in high-quality
       savings routes get a strong multiplicative boost, giving the ACO a
       more accurate structural signal than pure distance.
    3. Mild demand-aware bias: slightly prefer high-demand nodes to be visited
       earlier (when capacity is fresh), improving packing.
    4. Angular proximity bonus: prefer edges between nodes in similar angular
       sectors from the depot, reducing route crossings.
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    safe_dist = distance_matrix + eps
    
    # 1. Base heuristic: sharpened inverse distance (d^-3)
    base_heuristic = 1.0 / safe_dist ** 3
    
    # 2. Clarke-Wright savings computation
    # savings[i, j] = dist[0,i] + dist[0,j] - dist[i,j]
    # For each customer pair (i,j), the saving from merging two routes
    # (0->i) and (0->j) into (0->i->j) or (0->j->i).
    # We use this to boost edges that are part of good savings.
    # savings[i,j] for i,j >= 1:
    #   s(i,j) = d(0,i) + d(0,j) - d(i,j)
    # The larger the saving, the more attractive the edge i->j (or j->i) is.
    
    # Compute savings for all customer pairs
    # d0[i] = dist[0, i] for i in 1..n-1
    d0 = distance_matrix[0, 1:]  # shape (n-1,)
    
    # savings_matrix[i, j] for i, j in 1..n-1
    # = d0[i-1] + d0[j-1] - distance_matrix[i, j]
    d0_col = d0[:, np.newaxis]  # (n-1, 1)
    d0_row = d0[np.newaxis, :]  # (1, n-1)
    dist_cc = distance_matrix[1:, 1:]  # (n-1, n-1)
    savings_cc = d0_col + d0_row - dist_cc  # (n-1, n-1)
    
    # Clip savings to be non-negative (some pairs may have negative savings)
    savings_cc = np.maximum(savings_cc, 0.0)
    
    # Normalize savings to [0, 1] range for scaling
    max_savings = np.max(savings_cc) if savings_cc.size > 0 else 1.0
    if max_savings > 0:
        savings_norm = savings_cc / max_savings
    else:
        savings_norm = savings_cc
    
    # Build the full savings boost matrix
    # For customer-to-customer edges: boost = 1 + alpha_s * savings_norm[i,j]
    # For depot-to-customer and customer-to-depot: no savings boost (1.0)
    alpha_s = 2.0  # Strong boost for high-savings edges
    savings_boost_cc = 1.0 + alpha_s * savings_norm  # (n-1, n-1)
    
    savings_boost = np.ones((n, n))
    savings_boost[1:, 1:] = savings_boost_cc
    
    # 3. Demand-aware bias: prefer high-demand nodes slightly
    # This encourages visiting large items when capacity is fresh.
    max_demand = np.max(demands) if n > 1 else 1.0
    if max_demand > 0:
        demand_norm = demands / max_demand
    else:
        demand_norm = demands
    
    # Soft bias: 1 + 0.2 * demand_norm for edges TO customer j
    demand_factor = 1.0 + 0.2 * demand_norm
    demand_factor[0] = 1.0  # Depot has no demand
    demand_boost = np.ones((n, n))
    for j in range(1, n):
        demand_boost[:, j] = demand_factor[j]
    
    # 4. Angular proximity bonus
    # Compute angle from depot to each customer
    depot = coordinates[0]
    if n > 1:
        dx = coordinates[1:, 0] - depot[0]
        dy = coordinates[1:, 1] - depot[1]
        angles = np.arctan2(dy, dx)  # shape (n-1,)
        
        # Angular difference matrix for customer-to-customer edges
        angles_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
        angles_diff = (angles_diff + np.pi) % (2 * np.pi) - np.pi
        # cos of angle difference: 1 when same direction, -1 when opposite
        alpha_a = 0.3
        angular_bonus_cc = 1.0 + alpha_a * np.cos(angles_diff)  # (n-1, n-1)
        
        angular_bonus = np.ones((n, n))
        angular_bonus[1:, 1:] = angular_bonus_cc
    else:
        angular_bonus = np.ones((n, n))
    
    # Combine all components
    result = base_heuristic * savings_boost * demand_boost * angular_bonus
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    result = np.maximum(result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
