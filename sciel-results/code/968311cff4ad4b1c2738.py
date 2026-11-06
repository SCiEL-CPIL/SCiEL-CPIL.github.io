"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 140.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _select_node_density(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a local density heuristic.
    
    Instead of just looking at distance to current/destination, we estimate the
    'local cost' of adding a node. This is approximated by how well the node fits
    into the local neighborhood of the current node.
    
    We use a 'gap' metric:
    For each candidate c in unvisited:
    1. Find the nearest neighbor of c in the *visited* set (including current).
    2. Find the nearest neighbor of c in the *unvisited* set (excluding c).
    3. The 'density score' is high if c is close to visited nodes (good connection)
       but the 'gap' to unvisited is small (preserves future options).
       
    Actually, a simpler and faster proxy for "local density optimization" in constructive TSP:
    Minimize: d(current, c) + alpha * d(c, nearest_unvisited(c))
    
    This encourages moving to a node that is close to the current position AND
    close to other unvisited nodes (i.e., in a dense cluster), avoiding isolated
    points that would force long jumps later.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Vectorized distance from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from each unvisited to all other unvisited
    # d_unv[i, j] is distance between unvisited[i] and unvisited[j]
    d_unv = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # Find the minimum distance from each unvisited node to any *other* unvisited node
    # We need to ignore the self-distance (which is 0)
    np.fill_diagonal(d_unv, np.inf)
    min_dist_to_other_unvisited = np.min(d_unv, axis=1)
    
    # Heuristic:
    # We want to minimize d_curr (local step cost)
    # But we also want to stay in dense areas. 
    # If min_dist_to_other_unvisited is very small, the node is in a cluster.
    # If it's large, it's isolated.
    # Let's try to penalize isolated nodes.
    # Score = d_curr + beta * max(0, min_dist_to_other_unvisited - threshold)
    # Or simpler: Score = d_curr + beta * min_dist_to_other_unvisited
    
    # Let's try a weighted sum.
    # A node that is close to current AND close to others is best.
    # This is effectively a "nearest neighbor in a graph that includes current and unvisited".
    
    # Let's refine: 
    # The cost of adding c to the tour ending at current is d(current, c).
    # The future cost is hard to predict, but if c is isolated, the next step will be expensive.
    # So we estimate the "local tour length increase" as d(current, c) + d(c, nearest_neighbor_in_unvisited(c)).
    # This approximates the path current -> c -> next.
    
    alpha = 0.5
    scores = d_curr + alpha * min_dist_to_other_unvisited
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # Use density heuristic
    next_node = _select_node_density(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
