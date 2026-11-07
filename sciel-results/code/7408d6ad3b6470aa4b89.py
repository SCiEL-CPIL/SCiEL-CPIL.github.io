import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Structural change: Use spatial density-aware heuristic.
    
    Key insight: In CVRP, customers that are spatially close tend to be on the
    same route. The heuristic should reflect not just pairwise distance but
    also whether an edge is "route-consistent" — i.e., whether i and j are
    likely to be visited in the same tour segment.
    
    We use a two-part structural model:
    1. Exponential distance decay: exp(-d[i,j] / scale) instead of 1/d
       This gives a smoother, more bounded heuristic that doesn't blow up
       for very close nodes.
    2. Spatial density factor: For each node j, compute how many other nodes
       are within a radius proportional to capacity/demand[j]. Nodes in dense
       regions get a boost because they're likely to be on the same route.
    
    This is structurally different from inverse-distance × demand-factor
    because it captures *collective* spatial structure rather than just
    pairwise properties.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    d = distance_matrix.astype(float)
    n = d.shape[0]
    
    # Distance scale: typical inter-customer distance
    # Use median of off-diagonal distances as a natural scale
    off_diag = d[~np.eye(n, dtype=bool)]
    if len(off_diag) > 0:
        dist_scale = np.median(off_diag)
    else:
        dist_scale = 1.0
    if dist_scale < 1e-6:
        dist_scale = 1.0
    
    # Part 1: Exponential distance decay
    # eta_dist[i,j] = exp(-d[i,j] / dist_scale)
    # This is bounded in (0, 1] and smooth
    eta_dist = np.exp(-d / dist_scale)
    
    # Part 2: Spatial density factor for each node
    # For node j, count how many other customers are within radius r_j
    # where r_j is proportional to sqrt(capacity / demand[j]) / n
    # This approximates the "route neighborhood" size for node j
    
    # Compute radius for each node based on demand
    # Higher demand -> smaller neighborhood (fewer nodes can fit)
    # Lower demand -> larger neighborhood
    demands_float = demands.astype(float)
    # r_j = c * sqrt(capacity / demand_j) / sqrt(n)
    # Choose c so that typical radius is ~1/sqrt(n) of the unit square
    c = 1.0
    radii = c * np.sqrt(capacity / (demands_float + 1e-9)) / np.sqrt(n)
    # Cap radii to avoid unreasonable values
    radii = np.clip(radii, 0.05, 1.0)
    
    # Compute density: for each node j, count neighbors within r_j
    # Use coordinates (n, 2)
    coords = coordinates.astype(float)
    
    density = np.ones(n)
    for j in range(n):
        if j == 0:
            continue  # Depot
        # Distance from j to all other nodes
        dists_j = np.sqrt(np.sum((coords - coords[j]) ** 2, axis=1))
        # Count nodes within r_j (excluding self)
        count = np.sum(dists_j < radii[j])
        # Normalize: density is relative to expected random count
        # Expected count in random uniform distribution: pi * r^2 * (n-1)
        expected = np.pi * radii[j] ** 2 * (n - 1)
        if expected > 1e-6:
            density[j] = count / expected
        else:
            density[j] = 1.0
    
    # Depot gets average density
    density[0] = np.mean(density[1:]) if n > 1 else 1.0
    
    # Clip density to reasonable range
    density = np.clip(density, 0.5, 3.0)
    
    # Combine: eta[i,j] = eta_dist[i,j] * density[j]
    # The density factor boosts edges to nodes in dense regions
    density_col = density.reshape(1, -1)
    heuristic = eta_dist * density_col
    
    # Set diagonal to 0 (will be masked)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    # Check deadline
    if time.time() > deadline:
        pass  # We're already done
    
    return heuristic
