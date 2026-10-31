"""Rewritten program: MST-aware TSP constructive heuristic with degree-based penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 5 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: MST-aware heuristic. For each candidate, estimate the residual connectivity
    cost using a sum-of-nearest-neighbors proxy for MST, plus a destination-bias term.
    This captures global structure of the unvisited subgraph.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline with safety margin
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Compute base distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive parameters based on progress
    total_nodes = 50
    progress = 1.0 - (n_unvisited / total_nodes)
    
    # Beta: weight on residual connectivity (MST proxy)
    # Start higher when many nodes remain (global structure matters), lower near end
    beta = 0.08 * (1.0 - 0.5 * progress)
    
    # Gamma: weight on destination proximity
    # Increase as we approach the end to ensure cheap tour closure
    gamma = 0.05 + 0.35 * progress
    
    # If we have few nodes left, use simpler scoring
    if n_unvisited <= 8:
        scores = dists_from_current + gamma * dists_to_dest
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
    
    # For larger sets, compute MST proxy (sum of nearest neighbors) for residual graph
    # For each candidate u, removing it from unvisited, compute sum of NN for remaining nodes
    # This is O(n^2) which is acceptable for n=50
    
    unvisited_set = set(unvisited_nodes)
    
    # Precompute: for each node in unvisited, its nearest neighbor among unvisited
    # (excluding itself) - this is the "degree-1" contribution
    nn_distances = np.zeros(n_unvisited)
    for i in range(n_unvisited):
        node_i = unvisited_nodes[i]
        # Find min distance from node_i to any other unvisited node
        if n_unvisited > 1:
            others_mask = np.ones(n_unvisited, dtype=bool)
            others_mask[i] = False
            if np.any(others_mask):
                nn_distances[i] = np.min(distance_matrix[node_i, unvisited_nodes[others_mask]])
            else:
                nn_distances[i] = 0.0
        else:
            nn_distances[i] = 0.0
    
    # Total sum of NN for current unvisited set
    total_nn = np.sum(nn_distances)
    
    # For each candidate, the residual NN sum = total_nn - nn_distances[candidate]
    # (approximation: removing a node only affects its own NN term, not others' NN terms)
    # This is a first-order approximation that's fast
    residual_nn = total_nn - nn_distances
    
    # Compute scores
    scores = dists_from_current + beta * residual_nn + gamma * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
