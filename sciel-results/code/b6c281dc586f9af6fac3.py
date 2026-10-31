"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 5.0  # seconds of safety margin before 150s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > 150.0 - DEADLINE_MARGIN


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    
    Uses a blended heuristic:
    - Nearest neighbor component (greedy local)
    - Destination bias (pull toward the final destination to avoid long return)
    - Farthest-penalty (avoid going too far from the main cluster)
    
    The blend weights are tuned to balance local greediness with global tour quality.
    """
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    if _deadline_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from each unvisited to the destination node
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize distances for blending
    # Use a soft combination: score = w1 * (dist_from_current / max) + w2 * (dist_to_dest / max)
    # Lower score is better
    
    max_d1 = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
    max_d2 = np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0
    
    # Blend weights: emphasize nearest neighbor but pull toward destination
    # When few nodes remain, increase destination weight to ensure we can return
    remaining_fraction = n_unvisited / 50.0  # assume ~50 cities total
    w_dest = 0.3 + 0.4 * (1.0 - remaining_fraction)  # increases as we near the end
    w_nn = 1.0 - w_dest
    
    # Combined score (lower is better)
    scores = w_nn * (dist_from_current / max_d1) + w_dest * (dist_to_dest / max_d2)
    
    # Also add a small penalty for nodes that are very far from current to avoid jumps
    # Use a soft-max-like penalty
    far_threshold = np.mean(dist_from_current) + 0.5 * np.std(dist_from_current)
    far_penalty = np.where(dist_from_current > far_threshold, 
                           0.1 * (dist_from_current - far_threshold) / max_d1, 
                           0.0)
    scores += far_penalty
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
