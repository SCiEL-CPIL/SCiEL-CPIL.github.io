"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix (larger = more attractive) of the same shape.
    
    Combines:
    1. Spatial density field (node-potential): each node's attractiveness is modulated
       by the local density of demand in its neighborhood, encouraging ants to visit
       regions where multiple customers can be served efficiently.
    2. Demand-clustering pressure: edges between high-demand nodes are boosted to
       encourage serving them together (reducing fragmentation).
    3. Depot-anchored structure: edges from depot are weighted by demand efficiency,
       and edges back to depot are boosted for proximity.
    
    The structural idea: instead of a static hand-crafted formula, we compute a
    spatial density field that captures which regions of the unit square have
    concentrated demand, then use that to modulate edge attractiveness.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Ensure finite and positive distance matrix
    dm = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    dm = np.maximum(dm, 1e-9)
    
    # Base heuristic: inverse distance
    base = 1.0 / dm
    
    # --- Step 1: Compute spatial density field ---
    # For each customer node, compute the local demand density:
    # sum of demands of nearby customers weighted by Gaussian kernel
    # This identifies "dense" regions where clustering is beneficial
    
    customers = np.arange(1, n)
    cust_coords = coords[1:]
    cust_demands = dem[1:]
    
    # Compute pairwise distances between customers
    # Use Euclidean distance from coordinates for the density computation
    # (distance_matrix may already be Euclidean, but we use coords for consistency)
    diff = cust_coords[:, None, :] - cust_coords[None, :, :]
    dist_cust = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Gaussian kernel with bandwidth based on typical inter-customer distance
    if len(customers) > 1:
        typical_dist = np.mean(dist_cust[dist_cust > 0])
    else:
        typical_dist = 0.5
    bandwidth = max(typical_dist * 0.8, 0.05)
    
    # Density at each customer: weighted sum of demands in neighborhood
    # density[i] = sum_j demands[j] * exp(-dist(i,j)^2 / (2*bandwidth^2))
    kernel = np.exp(-dist_cust ** 2 / (2 * bandwidth ** 2))
    density = kernel @ cust_demands  # shape (n-1,)
    
    # Normalize density to [0.5, 2.0] range to modulate without dominating
    if density.max() > 0:
        density_norm = 0.5 + 1.5 * (density - density.min()) / (density.max() - density.min() + 1e-9)
    else:
        density_norm = np.ones(n - 1)
    
    # Extend density to full node vector (depot gets neutral value)
    density_full = np.ones(n)
    density_full[1:] = density_norm
    
    # --- Step 2: Demand-clustering pressure ---
    # Edges between nodes with high combined demand should be boosted
    # This encourages serving high-demand customers together
    demand_outer = np.outer(dem, dem)
    # Normalize by capacity to keep in reasonable range
    demand_pressure = demand_outer / cap
    
    # Only apply demand pressure to customer-to-customer edges
    # Scale to add a moderate boost (0.3x of base)
    customer_mask = np.zeros((n, n), dtype=bool)
    customer_mask[1:, 1:] = True
    
    demand_boost_matrix = np.zeros((n, n))
    demand_boost_matrix[customer_mask] = 0.3 * demand_pressure[customer_mask] * base[customer_mask]
    
    # --- Step 3: Depot-anchored structure ---
    # Depot to customer: boost by demand (high demand = visit early when truck is empty)
    depot_to_cust = np.zeros(n)
    if n > 1:
        depot_to_cust[1:] = 0.2 * (dem[1:] / max(dem[1:].max(), 1.0)) * base[0, 1:]
    
    # Customer to depot: boost by proximity (closer to depot = easier return)
    cust_to_depot = np.zeros(n)
    if n > 1:
        cust_to_depot[1:] = 0.15 * base[1:, 0]
    
    # --- Combine all components ---
    # heuristic[i,j] = base[i,j] * density_full[j] + demand_boost[i,j] + depot_adjustments
    heuristic = base * density_full[None, :]
    
    # Add demand clustering boost
    heuristic += demand_boost_matrix
    
    # Add depot structure
    heuristic[0, 1:] += depot_to_cust[1:]
    heuristic[1:, 0] += cust_to_depot[1:]
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Time guard: if we're running long, just return what we have
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
