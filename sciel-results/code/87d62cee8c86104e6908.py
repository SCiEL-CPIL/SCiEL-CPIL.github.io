"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}


def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Among capacity-feasible unvisited nodes, choose the one that
    minimizes a composite score combining:
      - distance from current node to candidate
      - a "continuation cost" estimate (distance from candidate to its nearest
        remaining feasible neighbor, or to depot if none)
    This encourages tight, clustered routes.
    
    If no feasible node exists, return to depot.
    """
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Compute distances from current node to all feasible nodes
    feasible_arr = np.array(feasible)
    dists_to_current = dm[current_node, feasible_arr]
    
    # For each feasible candidate, estimate the "continuation" cost:
    # After visiting candidate, what's the cheapest next step?
    # Approximate as: min distance from candidate to any other feasible node,
    # or distance to depot if no other feasible nodes remain.
    # This is O(|feasible|^2) but |feasible| <= 50, so at most 2500 ops.
    
    # We want to minimize: dist(current -> cand) + w * continuation(cand)
    # where continuation(cand) = min(dist(cand -> other_feasible), dist(cand -> depot))
    # But we should be careful: if only one node remains after picking cand, 
    # continuation should be dist(cand -> depot).
    
    n_feasible = len(feasible_arr)
    
    # Build sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    # For each candidate i, the best next step among other feasible nodes:
    # min over j != i of sub_dm[i,j], and also consider going to depot
    # dist_to_depot[i] = dm[feasible_arr[i], depot]
    dist_to_depot = dm[feasible_arr, depot]
    
    # For each i, continuation = min(min_{j!=i} sub_dm[i,j], dist_to_depot[i])
    # But if n_feasible == 1, continuation is just dist_to_depot[0]
    
    if n_feasible > 1:
        # Set diagonal to inf so we don't pick self
        sub_dm_copy = sub_dm.copy()
        np.fill_diagonal(sub_dm_copy, np.inf)
        min_to_other = np.min(sub_dm_copy, axis=1)
        continuation = np.minimum(min_to_other, dist_to_depot)
    else:
        continuation = dist_to_depot
    
    # Composite score: we want to minimize total route length.
    # A good heuristic: dist_to_current + alpha * continuation
    # alpha < 1 because continuation is an estimate of future cost,
    # and we want to prioritize getting close now.
    # Let's try alpha = 0.5 to balance immediate and future costs.
    
    alpha = 0.5
    scores = dists_to_current + alpha * continuation
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
