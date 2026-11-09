"""Combined CVRP heuristic: savings-based selection with look-ahead and capacity awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings-based heuristic with look-ahead and capacity-awareness.
    Returns 0 (depot) if no feasible node can be added, if savings are negative,
    or if the time limit is reached.
    
    Strategy:
    - When at depot, pick the nearest feasible node to start the route.
    - Otherwise, compute savings for each feasible node:
      saving(j) = dist(current, depot) + dist(depot, j) - dist(current, j)
    - Add a look-ahead term: estimate the cost of the next step after visiting j.
    - If max adjusted saving is negative, close the route (go to depot).
    - Otherwise, pick the node with maximum adjusted saving.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    n_feasible = len(feasible_nodes)
    
    # Compute distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    # Calculate savings for each feasible node
    savings = d_current_depot + d_depot_j - d_current_j
    
    # Look-ahead term: estimate the cost of the next step after visiting j
    # For each candidate j, estimate the minimum additional cost to either:
    # 1. Go to the nearest other feasible node k (where k != j) and then to depot
    # 2. Go directly to depot
    # We approximate this by: min over k != j of dist(j, k) + dist(k, depot) vs dist(j, depot)
    # But computing this for all j is O(n^2). Instead, we use a simpler heuristic:
    # The "continuation cost" after visiting j is roughly the minimum distance from j to any
    # other feasible node plus the average distance from that node to depot, OR just dist(j, depot).
    
    # Simpler look-ahead: penalize nodes that are far from other feasible nodes
    # (i.e., nodes that would force a long detour for the next visit)
    # Use the second-nearest distance as a proxy for continuation cost
    
    if n_feasible > 1:
        # For each j, find the minimum distance to any other feasible node k
        # dist_matrix[feasible_nodes, feasible_nodes] gives all pairwise distances
        dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        
        # Set diagonal to inf to exclude self
        np.fill_diagonal(dist_matrix_feasible, np.inf)
        
        # For each j, the nearest other node's distance
        nearest_other_dist = np.min(dist_matrix_feasible, axis=1)
        
        # Look-ahead penalty: if the nearest other node is far, it's more expensive to continue
        # We add a fraction of this to the cost (subtract from savings)
        # Weight: 0.3 seems reasonable - not too aggressive
        look_ahead_penalty = 0.3 * nearest_other_dist
    else:
        # Only one feasible node: continuation cost is just going to depot
        look_ahead_penalty = np.zeros(1)
    
    # Adjusted savings: subtract look-ahead penalty
    adjusted_savings = savings - look_ahead_penalty
    
    # If the maximum adjusted saving is negative, it is cheaper to close the route
    if np.max(adjusted_savings) < 0:
        return int(depot)
    
    # Pick the node with the maximum adjusted saving
    max_idx = np.argmax(adjusted_savings)
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
