"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic attractiveness matrix for ACO on CVRP.
    
    Structural change: Instead of a pure distance-based heuristic with depot
    penalties, we use a demand-aware heuristic that considers:
    1. Distance-based attraction (1/d^1.5 - moderate exponent)
    2. Demand-based adjustment: prefer customers with demand that fits well
       with remaining capacity (approximated by favoring moderate demands)
    3. Spatial clustering: use coordinates to identify clusters and boost
       intra-cluster edges
    
    The key insight is that for CVRP, the best routes tend to serve geographically
    close customers together. We encode this by computing a "cluster affinity"
    based on local density.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    epsilon = 1e-9
    
    # Base: inverse distance with moderate exponent
    eta = 1.0 / ((d + epsilon) ** 1.5)
    
    # Compute local density for each customer (number of nearby customers)
    # Use a simple kernel: count customers within radius 0.15
    radius = 0.15
    density = np.zeros(n)
    for i in range(1, n):
        dists = np.linalg.norm(coords[1:n] - coords[i], axis=1)
        density[i] = np.sum(dists < radius)
    
    # Normalize density to [0, 1]
    max_density = np.max(density) if np.max(density) > 0 else 1
    density_norm = density / max_density
    
    # Boost edges between customers in dense areas (encourages serving clusters)
    # For customer-to-customer edges, boost by average density of endpoints
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                avg_density = (density_norm[i] + density_norm[j]) / 2.0
                eta[i, j] *= (1.0 + 0.5 * avg_density)
    
    # Depot-to-customer edges: boost customers in dense areas
    for j in range(1, n):
        eta[0, j] *= (1.0 + 0.3 * density_norm[j])
        eta[j, 0] *= (1.0 + 0.3 * density_norm[j])
    
    # Demand adjustment: slightly penalize very high demand customers 
    # (they fill the truck quickly, potentially leading to shorter routes)
    # and slightly boost moderate demand
    demand_factor = np.ones(n)
    for i in range(1, n):
        dem = demands_arr[i]
        # Optimal demand is around capacity/3 to capacity/2 for a single stop
        # But since we're doing multiple stops, moderate is better
        if dem > 6:
            demand_factor[i] = 0.85
        elif dem < 2:
            demand_factor[i] = 0.95
        else:
            demand_factor[i] = 1.0
    
    # Apply demand factor to edges involving customer i
    for i in range(1, n):
        eta[i, :] *= demand_factor[i]
        eta[:, i] *= demand_factor[i]
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
