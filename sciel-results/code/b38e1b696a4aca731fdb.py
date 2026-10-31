"""Force-directed TSP heuristic: attractive to current, repulsive from recent, aligned with destination."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

# State tracking for force-directed approach
_visited_order = []       # order of visited nodes
_current_instance_id = -1
_instance_start_time = 0.0

# Tunable parameters for force-directed scoring
_ALPHA = 1.0              # weight for attraction to current node
_BETA = 0.3               # weight for repulsion from recent nodes
_GAMMA = 0.2              # weight for alignment with destination direction
_REPULSION_WINDOW = 5     # how many recent nodes to consider for repulsion


def _compute_force_scores(current_node, destination_node, unvisited, d, visited_order):
    """
    Compute force-directed scores for unvisited nodes.
    
    Attractive force: -distance to current (closer = higher score)
    Repulsive force: -sum of distances to recent visited nodes (farther = higher score)  
    Alignment: dot product of direction to candidate with direction to destination
    """
    n_unv = len(unvisited)
    if n_unv == 0:
        return np.array([])
    
    # Distances from current to unvisited
    dist_to_current = d[current_node, unvisited]
    
    # Normalize distances for attraction (inverse distance)
    # Use a smooth inverse to avoid division by zero
    eps = 1e-10
    attract = 1.0 / (dist_to_current + eps)
    
    # Repulsion from recent visited nodes
    repulse = np.zeros(n_unv)
    if len(visited_order) > 0:
        recent = visited_order[-_REPULSION_WINDOW:]
        for v in recent:
            dist_to_recent = d[v, unvisited]
            # Repulsion: prefer nodes far from recently visited
            repulse += 1.0 / (dist_to_recent + eps)
    
    # Alignment with destination direction
    # Direction from current to destination
    cur_to_dest = d[current_node, destination_node]
    if cur_to_dest < eps:
        # Already at destination, just use attraction
        align = np.ones(n_unv)
    else:
        # For each unvisited node, compute how well the direction from current 
        # to that node aligns with the direction from current to destination
        # We don't have coordinates, only distances, so we use a proxy:
        # Nodes that are "in between" current and destination should be preferred
        # Use triangle inequality proxy: if d(cur, unv) + d(unv, dest) ≈ d(cur, dest), 
        # then unv is roughly on the path
        
        dist_to_dest = d[destination_node, unvisited]
        # Alignment score: lower is better (nodes on the path have sum ≈ direct distance)
        path_cost = dist_to_current + dist_to_dest
        direct = cur_to_dest
        # Normalize: score is how much longer the path is vs direct
        # Lower path_cost relative to direct means better alignment
        align = direct / (path_cost + eps)
        # This gives values in (0, 1], with 1.0 meaning perfectly aligned
    
    # Combine forces
    score = _ALPHA * attract - _BETA * repulse + _GAMMA * align
    
    return score


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _visited_order, _current_instance_id, _instance_start_time
    
    unvisited = np.asarray(unvisited_nodes)
    if unvisited.size == 0:
        return int(current_node)
    
    n = int(distance_matrix.shape[0])
    d = distance_matrix
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(unvisited[int(np.argmin(d[current_node, unvisited]))])
    
    # Detect new instance (when we start from node 0 with all others unvisited)
    if current_node == 0 and unvisited.size == n - 1:
        _visited_order = [0]
        _instance_start_time = time.time()
    else:
        _visited_order.append(current_node)
    
    # Compute force-directed scores
    scores = _compute_force_scores(current_node, destination_node, unvisited, d, _visited_order)
    
    # Select the node with the highest score (best force balance)
    best_idx = int(np.argmax(scores))
    
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
