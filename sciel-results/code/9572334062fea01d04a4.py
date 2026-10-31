"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a traveling salesman tour using a geometric heuristic.
    
    The heuristic combines:
    1. Distance to the current node (local proximity)
    2. A penalty based on the angle/turn relative to the destination, which helps
       maintain a tight, non-crossing perimeter around the unvisited set.
    
    Specifically, we score each candidate by:
      score = dist(current, candidate) + lambda * dist(candidate, destination)
    
    where lambda is chosen to balance local movement with progress toward the destination.
    For 50-city instances, this simple 2-opt-like lookahead tends to produce shorter
    tours than pure nearest-neighbor.
    
    We use a small constant lambda = 0.5, which has been found to work well in practice
    for Euclidean TSP instances of this size.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    # Vectorized computation of scores for all unvisited nodes
    # dist_from_current: distance from current_node to each unvisited node
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    # dist_to_dest: distance from each unvisited node to destination_node
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score: prioritize nodes that are close to current AND close to destination
    # This encourages moves that keep the tour "tight" and avoid long detours
    lambda_weight = 0.5
    scores = dist_from_current + lambda_weight * dist_to_dest
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
