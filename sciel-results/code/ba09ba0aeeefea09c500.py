# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic edge-attractiveness matrix for ant colony optimization
    of the capacitated vehicle routing problem.
    
    Uses a structural heuristic based on:
    1. Inverse distance (base attractiveness)
    2. Demand compatibility (edges between high-demand nodes are less attractive
       if they would likely exceed capacity when combined with other nodes)
    3. Proximity to depot (nodes close to depot get bonus for return trips)
    4. Local density (edges in dense regions are slightly more attractive)
    
    The heuristic is static, computed once per instance.
    """
    n = len(demands)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Ensure shape
    if dm.shape != (n, n):
        raise ValueError("distance_matrix shape mismatch")
    
    # Base: inverse distance with small epsilon to avoid division by zero
    eps = 1e-9
    base = 1.0 / (dm + eps)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(base, 0.0)
    
    # Demand compatibility factor:
    # For each pair (i, j), if demand[i] + demand[j] > capacity * 0.6,
    # they are less likely to be on the same route, so reduce attractiveness.
    # This is a soft penalty.
    demand_sum = dem[:, None] + dem[None, :]
    # Scale: if sum > 0.6 * cap, penalize
    threshold = 0.6 * cap
    # Penalty factor: 1.0 if sum <= threshold, decreasing linearly to 0.3 at 2*threshold
    penalty = np.ones_like(demand_sum)
    over = demand_sum > threshold
    if np.any(over):
        # Linear decay from 1.0 at threshold to 0.3 at 2*threshold
        excess = demand_sum[over] - threshold
        max_excess = threshold  # so 2*threshold is max
        decay = 1.0 - 0.7 * np.clip(excess / max_excess, 0, 1)
        penalty[over] = decay
    
    # Depot proximity bonus:
    # Nodes closer to depot (index 0) get a slight boost on edges involving depot
    # This helps ants prefer returning to depot from nearby nodes
    dist_to_depot = dm[0, :]  # distance from each node to depot
    min_dist = dist_to_depot.min()
    max_dist = dist_to_depot.max()
    if max_dist > min_dist:
        norm_dist = (dist_to_depot - min_dist) / (max_dist - min_dist)
    else:
        norm_dist = np.zeros(n)
    
    # Depot row/column bonus: edges to/from depot get boosted if node is close
    depot_bonus = np.ones((n, n))
    # For edges from depot to node i: bonus based on closeness
    close_factor = 1.0 + 0.5 * (1.0 - norm_dist)  # 1.0 to 1.5
    depot_bonus[0, :] = close_factor
    depot_bonus[:, 0] = close_factor
    np.fill_diagonal(depot_bonus, 1.0)
    
    # Local density bonus:
    # For each node, compute average distance to its k nearest neighbors
    # Edges to nearby nodes (within local cluster) get a small boost
    k = min(5, n - 1)
    local_density = np.ones(n)
    for i in range(n):
        if i == 0:
            continue
        dists_i = dm[i, :].copy()
        dists_i[i] = np.inf
        nn_dists = np.sort(dists_i)[:k]
        avg_nn = nn_dists.mean()
        # Lower avg_nn means denser -> higher attractiveness
        local_density[i] = 1.0 + 0.3 * (1.0 - avg_nn / (max_dist + eps))
    
    # Apply local density to edges
    # For edge (i, j), multiply by average of local_density[i] and local_density[j]
    density_factor = (local_density[:, None] + local_density[None, :]) / 2.0
    
    # Combine all factors
    heuristic = base * penalty * depot_bonus * density_factor
    
    # Ensure positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Normalize to a reasonable range to avoid numerical issues
    # Scale so that the median is around 1.0
    med = np.median(heuristic[heuristic > 0])
    if med > 0:
        heuristic = heuristic / med
    
    # Final safety: ensure all positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
