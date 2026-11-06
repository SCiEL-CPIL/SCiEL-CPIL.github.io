# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute and cache useful structures per instance (keyed by id of distance_matrix)
# Since the evaluator calls select_next_node many times per instance, we cache per-instance data.

_cache = {}

def _get_cache_key(distance_matrix):
    return id(distance_matrix)

def _prepare(distance_matrix, demands, n):
    """
    Prepare per-instance data:
    - distance_matrix as numpy array
    - demands as numpy array
    - For each node, sorted list of other nodes by distance
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    d = np.asarray(demands, dtype=np.float64)
    
    # For each node, get sorted distances to all other nodes
    # sorted_neighbors[i] = list of (node, dist) sorted by dist
    sorted_neighbors = []
    for i in range(n):
        dists = dm[i].copy()
        dists[i] = np.inf  # exclude self
        order = np.argsort(dists)
        sorted_neighbors.append(order)
    
    return dm, d, sorted_neighbors

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Regret-based selection:
    - Filter feasible customers (demand <= rest_capacity)
    - If none feasible, return depot
    - For each feasible customer, compute regret:
      regret(j) = dist(current, second_best) - dist(current, best)
      where best and second_best are the two closest feasible customers from current
    - Actually, regret is typically: for node j, the cost of NOT visiting j now
      = min over other feasible k of (dist(current,j) + dist(j,k)) - dist(current,j)
      But simpler: regret(j) = dist(current, j) - min_{k != j, feasible} dist(current, k)
      This is the penalty for not going to the nearest feasible node.
      
    Let me use a clearer regret formulation:
    For each feasible customer j, define:
      best_dist = min distance from current to any feasible customer
      second_best_dist = second min distance from current to any feasible customer
    
    The regret of visiting j is:
      If j is the best: regret = second_best_dist - best_dist (low regret, it's the obvious choice)
      If j is not the best: regret = best_dist - dist(current, j) ... no that's negative.
    
    Standard regret-2:
      For each unvisited node j, let best_j be the closest feasible node to j (excluding j itself? no, including)
      Actually regret-2 for node j: 
        r1(j) = min distance from j to any feasible node (including j? no, to other nodes)
        r2(j) = second min distance from j to any feasible node
        regret(j) = r2(j) - r1(j)
      Then pick the node with highest regret.
    
    But we're choosing the NEXT node to visit from current_node. So the regret should be from the perspective of the current node.
    
    Let me use: for each feasible customer j, 
      regret(j) = (dist from current to j) - (min over k != j feasible of dist from current to k)
      This is negative for the best, positive... no.
    
    Simpler and effective: use a combined score that balances:
    1. Distance from current to j (closer is better)
    2. Regret: how "unique" j is as a good next stop
    
    Let me implement regret-2 properly:
    For each feasible customer j:
      Look at distances from j to all other feasible customers
      r1 = min distance, r2 = second min distance
      regret(j) = r2 - r1
    
    Pick the feasible j with highest regret. Tie-break by distance from current.
    """
    key = _get_cache_key(distance_matrix)
    n = len(demands)
    
    if key not in _cache:
        _cache[key] = _prepare(distance_matrix, demands, n)
        # Also cache a version counter to avoid stale data
        _cache_version = 0
    dm, d, sorted_neighbors = _cache[key]
    
    # Convert unvisited to numpy array
    unv = np.asarray(unvisited_nodes, dtype=np.int64)
    if len(unv) == 0:
        return depot
    
    # Filter feasible: demand <= rest_capacity
    feasible_mask = d[unv] <= rest_capacity
    feasible = unv[feasible_mask]
    
    if len(feasible) == 0:
        return depot
    
    # If only one feasible, go there
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Compute regret for each feasible customer
    # For each feasible j, find r1 and r2 distances from j to other feasible customers
    # Use sorted_neighbors to quickly find nearest feasible neighbors
    
    regrets = np.zeros(len(feasible))
    dists_from_current = dm[current_node, feasible]
    
    for idx, j in enumerate(feasible):
        # Get sorted neighbors of j
        nbrs = sorted_neighbors[j]
        # Find first two feasible neighbors (not j itself, which is excluded in sorted_neighbors)
        r1 = np.inf
        r2 = np.inf
        count = 0
        for nb in nbrs:
            if nb in feasible:
                if count == 0:
                    r1 = dm[j, nb]
                    count = 1
                elif count == 1:
                    r2 = dm[j, nb]
                    count = 2
                    break
            # We only need two, but we need to check membership
            # This is O(n) per node, which is slow. Let's optimize.
        # Actually, let's precompute a feasible set for faster lookup
        # But for now, let's try a vectorized approach
    
    # Better approach: for each feasible j, compute distances to all feasible nodes
    # Then find min and second min
    # This is O(k^2) where k = number of feasible nodes
    # k can be up to 50, so k^2 = 2500, which is fine
    
    feasible_set = set(feasible.tolist())
    for idx, j in enumerate(feasible):
        # Distances from j to all feasible nodes
        dists_to_feasible = dm[j, feasible]
        # j is in feasible, so dists_to_feasible includes 0 for j itself
        # Find the two smallest non-zero distances
        sorted_dists = np.sort(dists_to_feasible)
        # The first element should be 0 (distance to self)
        # r1 = sorted_dists[1], r2 = sorted_dasts[2]
        if len(sorted_dists) >= 3:
            r1 = sorted_dists[1]
            r2 = sorted_dists[2]
        elif len(sorted_dists) == 2:
            r1 = sorted_dists[1]
            r2 = r1  # Only one other node
        else:
            r1 = 0
            r2 = 0
        regrets[idx] = r2 - r1
    
    # Combine regret with distance from current
    # Higher regret is better, lower distance is better
    # Normalize and combine
    max_regret = np.max(regrets)
    if max_regret > 1e-10:
        normalized_regret = regrets / max_regret
    else:
        normalized_regret = np.zeros_like(regrets)
    
    max_dist = np.max(dists_from_current)
    if max_dist > 1e-10:
        normalized_dist = dists_from_current / max_dist
    else:
        normalized_dist = np.zeros_like(dists_from_current)
    
    # Score: higher regret is better, lower distance is better
    # score = normalized_regret - 0.5 * normalized_dist
    score = normalized_regret - 0.5 * normalized_dist
    
    best_idx = np.argmax(score)
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
