"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Strategy: Use a multi-factor heuristic that combines:
    1. Inverse distance (standard)
    2. Demand-awareness: prefer edges to customers with demand that fits well
       in remaining capacity (i.e., avoid edges to very large demand customers
       when we're already loaded, but since this is static, we use average demand fit)
    3. Angular proximity: prefer customers that are in a similar direction from depot
       (reduces route fragmentation)
    
    The heuristic is static (computed once), so we encode global routing structure.
    """
    n = distance_matrix.shape[0]
    dm = distance_matrix.astype(np.float64)
    
    # Ensure no zeros cause issues
    dm_safe = np.where(dm < 1e-9, 1e-9, dm)
    
    # Base: inverse distance
    base = 1.0 / dm_safe
    
    # Angular proximity heuristic:
    # For each pair (i, j), compute how well j fits in a "direction" from i
    # This encourages ants to build routes that go in roughly one direction
    # before returning, reducing total distance
    
    # Compute angles from each node to all other nodes
    # For node i, angle to node j = atan2(dy, dx)
    # We want to favor edges where the angle from depot->i to i->j is small
    # (i.e., continuing in roughly the same direction)
    
    # Simpler approach: for each customer j, compute its "sector" based on
    # angle from depot. Customers in the same sector should be connected.
    
    # Compute angle of each node from depot (node 0)
    dx = coordinates[:, 0] - coordinates[0, 0]
    dy = coordinates[:, 1] - coordinates[0, 1]
    angles = np.arctan2(dy, dx)  # -pi to pi
    
    # For each pair (i, j) where i != 0 and j != 0, compute angular difference
    # Favor small angular differences (same sector)
    # Also favor edges from depot to any customer
    
    # Create angular compatibility matrix
    ang_diff = np.abs(np.subtract.outer(angles, angles))
    # Wrap to [0, pi]
    ang_diff = np.minimum(ang_diff, 2 * np.pi - ang_diff)
    
    # Angular factor: higher for small angular differences
    # Use a smooth kernel: exp(-ang_diff^2 / sigma^2)
    sigma = 1.0  # radians
    angular_factor = np.exp(-ang_diff ** 2 / (2 * sigma ** 2))
    
    # For depot edges, angular factor should be 1 (no penalty)
    angular_factor[0, :] = 1.0
    angular_factor[:, 0] = 1.0
    np.fill_diagonal(angular_factor, 1.0)
    
    # Demand-awareness: 
    # Customers with demand close to capacity should be visited early (from depot)
    # Customers with small demand can be visited anywhere
    # Since heuristic is static, we use: for edge i->j, if j has high demand,
    # it's better to come from depot (low load). But we can't know load statically.
    # Instead, use: prefer edges to customers whose demand is moderate
    # (not too large, not too small) as they pack well.
    
    # Alternative: use demand to weight how "valuable" an edge is
    # Higher demand customer -> edge to it is more important
    demand_factor = 1.0 + demands / capacity  # scales 1.0 to 1.18
    
    # Combine: base * angular_factor * demand_factor
    heuristic = base * angular_factor * demand_factor[:, np.newaxis]
    
    # Ensure depot row/col are reasonable
    # For depot->customer, we already have base distance and angular=1
    # For customer->depot, same
    
    # Clip to positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
