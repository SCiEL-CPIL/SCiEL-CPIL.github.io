"""Improved TSP constructive heuristic with phase-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for per-instance state
_instance_cache = {}

def _get_instance_id(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Create a lightweight key to identify the instance."""
    # Use the shape and a hash of some elements to identify the instance
    n = distance_matrix.shape[0]
    # Use destination_node and the sorted unvisited nodes as part of the key
    return (n, tuple(sorted(unvisited_nodes.tolist())), destination_node)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Phase-aware heuristic for TSP tour construction.
    
    Strategy:
    - Early phase (most nodes unvisited): pure nearest neighbor
    - Late phase (few nodes left): blend in destination proximity to avoid long closing edge
    - Use a smooth transition rather than a hard switch
    
    The key insight is that pure NN often leaves one very long edge at the end.
    By gradually incorporating the destination (the start node, which is the last
    node to be visited before closing the tour) into the scoring, we can avoid
    this while still benefiting from NN's good local choices early on.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from all unvisited nodes to destination (start node)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Determine the phase based on how many nodes remain
    # Total nodes in the tour is approximately n_unvisited + 1 (current) + 1 (destination)
    # We use the fraction of remaining nodes to determine blending weight
    total_nodes = distance_matrix.shape[0]
    fraction_remaining = n_unvisited / total_nodes
    
    # Blend weight: 0 means pure NN, 1 means pure destination proximity
    # We want to start with pure NN and gradually increase destination influence
    # Use a smooth curve: for the first ~70% of nodes, stay mostly NN
    # For the last ~30%, gradually increase destination weight
    
    if fraction_remaining > 0.7:
        # Early phase: pure nearest neighbor
        weight_dest = 0.0
    elif fraction_remaining > 0.3:
        # Middle phase: gradually increase destination weight
        # Linear interpolation from 0 to 0.5
        t = (0.7 - fraction_remaining) / 0.4  # 0 to 1 as fraction goes from 0.7 to 0.3
        weight_dest = 0.5 * t
    else:
        # Late phase: stronger destination influence
        # Linear interpolation from 0.5 to 1.0
        t = (0.3 - fraction_remaining) / 0.3  # 0 to 1 as fraction goes from 0.3 to 0
        weight_dest = 0.5 + 0.5 * t
    
    # Combine scores: lower is better
    # score = dist_from_current + weight_dest * dist_to_dest
    # But we need to normalize so that the two terms are comparable
    # Use the mean of each to normalize
    
    if n_unvisited > 1:
        mean_dist_current = np.mean(dist_from_current)
        mean_dist_dest = np.mean(dist_to_dest)
        
        # Normalize to comparable scales
        norm_dist_current = dist_from_current / (mean_dist_current + 1e-10)
        norm_dist_dest = dist_to_dest / (mean_dist_dest + 1e-10)
        
        # Combined score
        scores = norm_dist_current + weight_dest * norm_dist_dest
    else:
        return int(unvisited_nodes[0])
    
    # Pick the node with minimum combined score
    idx = np.argmin(scores)
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
