"""Traveling salesman constructive heuristic with phase-aware blending and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track time and enable phase-aware heuristics
_start_time = None
_total_nodes = None
_nodes_visited = 0
_destination = None


def _reset_state(total_nodes, dest):
    global _start_time, _total_nodes, _nodes_visited, _destination
    if _start_time is None:
        _start_time = time.time()
    _total_nodes = total_nodes
    _nodes_visited = 0
    _destination = dest


def _increment_visited():
    global _nodes_visited
    _nodes_visited += 1


def _time_remaining():
    if _start_time is None:
        return 120.0
    return 120.0 - (time.time() - _start_time)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour construction.
    
    Uses a phase-aware heuristic:
    - Early phase: primarily nearest neighbor to build a tight local tour
    - Late phase: blend in destination proximity to avoid a long closing edge
    - Includes a time guard to ensure we finish within the deadline
    
    Parameters:
    - current_node: int, the current node
    - destination_node: int, the final destination (start node for closed tour)
    - unvisited_nodes: np.ndarray, array of unvisited node indices
    - distance_matrix: np.ndarray, pairwise distance matrix
    
    Returns:
    - int: the index of the next node to visit
    """
    global _start_time, _total_nodes, _nodes_visited, _destination
    
    # Initialize state if this is the first call
    if _start_time is None:
        _start_time = time.time()
        _total_nodes = len(distance_matrix)
        _nodes_visited = 0
        _destination = destination_node
    
    # Increment visited count
    _nodes_visited += 1
    
    # Check if we're running out of time - fall back to simple nearest neighbor
    if _time_remaining() < 5.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) <= 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited nodes to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Determine the phase based on how many nodes have been visited
    # For 50 cities, we visit nodes 1-50 (node 0 is typically the start)
    # So unvisited count goes from ~49 down to 1
    total = _total_nodes if _total_nodes is not None else 50
    remaining = len(unvisited_nodes)
    
    # Phase blending:
    # - When many nodes remain (early), weight toward nearest neighbor
    # - When few nodes remain (late), weight toward destination proximity
    # 
    # Use a smooth transition. For 50 nodes:
    # - remaining > 40: pure nearest neighbor (weight 0.0 for dest)
    # - remaining 20-40: gradually increase destination weight
    # - remaining < 20: strong destination preference (weight up to 0.7)
    
    if remaining > 40:
        dest_weight = 0.0
    elif remaining > 20:
        # Linear ramp from 0.0 to 0.5
        dest_weight = 0.5 * (40 - remaining) / 20.0
    else:
        # Linear ramp from 0.5 to 0.7
        dest_weight = 0.5 + 0.2 * (20 - remaining) / 20.0
    
    # Combine scores: lower is better
    # Normalize distances to [0, 1] range for fair blending
    max_dist_current = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
    max_dist_dest = np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0
    
    norm_current = dist_from_current / max_dist_current
    norm_dest = dist_to_dest / max_dist_dest
    
    # Combined score: (1 - dest_weight) * normalized current distance + dest_weight * normalized dest distance
    combined_score = (1.0 - dest_weight) * norm_current + dest_weight * norm_dest
    
    # Select the node with minimum combined score
    next_idx = np.argmin(combined_score)
    
    return int(unvisited_nodes[next_idx])
# EVOLVE-BLOCK-END
