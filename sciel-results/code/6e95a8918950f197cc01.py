"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Hybrid Scoring using Centroid Attraction and Savings.
    1. Filter feasible nodes.
    2. Compute centroid of feasible nodes.
    3. Score each candidate based on:
       - Distance from current node (local continuity)
       - Distance from centroid (cluster compactness)
       - Savings potential (merging with other nodes)
       - Distance to depot (closing route when capacity is low)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to simple greedy
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    
    # 1. Compute Centroid of feasible nodes
    # Using simple mean of coordinates? No, we only have distance matrix.
    # We can approximate "centroid" in distance space or just use the node
    # that minimizes the sum of distances to all other feasible nodes.
    # Let's find the "geometric center" proxy: node k that minimizes sum(d(k, j) for j in feasible)
    # This is O(N^2) which is fine for N=50.
    
    # Sum of distances from each feasible node to all other feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    dist_sums = np.sum(sub_dm, axis=1)
    centroid_idx = np.argmin(dist_sums)
    centroid_node = feasible_arr[centroid_idx]
    
    # Distances from each candidate to the centroid node
    dist_to_centroid = dm[feasible_arr, centroid_node]
    
    # 2. Compute Savings Potential
    # Savings of linking i and j: d(depot, i) + d(depot, j) - d(i, j)
    # For each i, find the best j != i that maximizes savings.
    dists_to_depot = dm[feasible_arr, depot]
    
    # Savings matrix: savings[i, j] = d(depot, i) + d(depot, j) - d(i, j)
    savings_matrix = dists_to_depot[:, None] + dists_to_depot[None, :] - sub_dm
    # Set diagonal to -inf so we don't pick self
    np.fill_diagonal(savings_matrix, -np.inf)
    
    # Max savings for each node i
    max_savings = np.max(savings_matrix, axis=1)
    
    # Normalize savings to [0, 1] for weighting
    max_s_val = np.max(np.abs(max_savings))
    if max_s_val > 0:
        norm_savings = max_savings / max_s_val
    else:
        norm_savings = np.zeros_like(max_savings)
    
    # 3. Distance to Depot
    # We want to return to depot if capacity is low.
    # Estimate total capacity used or just use rest_capacity ratio.
    # Let's assume total demand is roughly sum(demands[unvisited]) + demands[current_node] (if not depot)
    # Simpler: Use a fixed weight that increases as rest_capacity decreases relative to max possible.
    # Max demand is 9. Min is 1.
    # If rest_capacity < 10, we are likely to close soon.
    
    # Adaptive weight for depot distance
    # If rest_capacity is small, we should prefer nodes closer to depot.
    # Let's define a threshold.
    cap_weight = 0.0
    if rest_capacity < 15:
        # Linearly increase weight as capacity drops
        cap_weight = 1.0 - (rest_capacity / 15.0)
        cap_weight = np.clip(cap_weight, 0.0, 1.0)
    
    # 4. Calculate Final Scores
    # Minimize: dist_current + w_centroid * dist_centroid - w_savings * savings + w_depot * dist_depot
    
    dists_from_current = dm[current_node, feasible_arr]
    
    # Weights
    w_centroid = 0.3
    w_savings = 0.5
    w_depot = cap_weight * 0.8
    
    scores = (dists_from_current 
              + w_centroid * dist_to_centroid 
              - w_savings * norm_savings 
              + w_depot * dists_to_depot)
    
    best_idx = np.argmin(scores)
    best_node = int(feasible_arr[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
