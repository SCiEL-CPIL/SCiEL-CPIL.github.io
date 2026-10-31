"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Nearest Neighbor heuristic with lookahead.
    
    This approach is a greedy local search. It primarily selects the closest
    unvisited city. However, to avoid getting trapped in a "dead end" where
    the remaining cities are far away, it uses a 1-step lookahead: it estimates
    the cost of moving to candidate j and then to the nearest remaining city k.
    This helps balance local proximity with future connectivity.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        # Cost if we go to n1 then n2 then destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 then n1 then destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    # Get distances from current_node to all unvisited nodes
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Find the index of the nearest neighbor in the unvisited array
    # This is the primary greedy choice
    nearest_idx_in_unvisited = np.argmin(dists)
    nearest_node = int(unvisited_nodes[nearest_idx_in_unvisited])
    
    # Lookahead: Check if a slightly farther node leads to a much better next step
    # We compare the cost of:
    # 1. Going to nearest_node, then to its nearest neighbor (among remaining)
    # 2. Going to a candidate node j, then to its nearest neighbor
    
    # To keep it efficient, we only do lookahead if n_unvisited is small enough
    # or we can just do it for all, as it's O(n^2) which is fine for n=50.
    # Actually, O(n^2) per step is 2500 ops. 50 steps -> 125k ops. Very fast.
    
    best_node = nearest_node
    best_score = dists[nearest_idx_in_unvisited]
    
    # We need to consider the "future" cost.
    # A simple heuristic: score(j) = dist(current, j) + 0.5 * min_dist(j, remaining\{j})
    # The 0.5 weight is a heuristic factor.
    
    for i, node in enumerate(unvisited_nodes):
        d_curr_to_node = dists[i]
        
        # Calculate min distance from this node to other unvisited nodes
        # We need to exclude the node itself
        other_nodes = np.concatenate((unvisited_nodes[:i], unvisited_nodes[i+1:]))
        if len(other_nodes) == 0:
            # Should not happen since n_unvisited > 2
            continue
            
        d_node_to_others = distance_matrix[node, other_nodes]
        min_d_next = np.min(d_node_to_others)
        
        # Heuristic score: immediate cost + discounted future cost
        # If this is the second to last node, the "future" is the return to destination?
        # No, the function just picks the next node. The evaluator handles the rest.
        # But if we are close to the end, we should be careful.
        # For n_unvisited > 2, the "next" step is just to another unvisited node.
        
        score = d_curr_to_node + 0.5 * min_d_next
        
        if score < best_score:
            best_score = score
            best_node = node

    return best_node
# EVOLVE-BLOCK-END
