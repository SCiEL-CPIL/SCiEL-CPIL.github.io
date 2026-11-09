"""Rewritten TSP constructive heuristic with 2-opt-style insertion look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0 - 5.0  # 5s safety margin

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a 2-opt-style insertion cost heuristic.
    
    For each candidate node u in unvisited_nodes, we estimate the cost of inserting u
    into the partial tour. Since we're building a path (not a cycle yet), the insertion
    cost for adding u after current_node is approximately:
        d(current_node, u) + d(u, next_unvisited)
    where next_unvisited is the node that would likely follow u (e.g., the nearest unvisited
    to u, or the destination if we're near the end).
    
    We minimize this insertion cost to avoid creating long edges that would be hard to
    fix later.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current_node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate u, estimate the "outgoing" cost: distance from u to the 
    # nearest other unvisited node (excluding itself), or to destination if few remain.
    # This models the edge that will be created after u is visited.
    
    # Precompute: for each unvisited node, the min distance to any other unvisited node
    # We use the distance matrix rows for unvisited nodes
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i (candidate u_i), find min distance to any j != i
    # Set diagonal to inf, then take row min
    np.fill_diagonal(unvisited_dist_matrix, np.inf)
    outgoing_cost = np.min(unvisited_dist_matrix, axis=1)
    
    # If only 2 unvisited nodes remain, the outgoing cost for each is the distance to the other
    # which is already handled correctly above.
    
    # Total insertion cost for candidate u:
    #   d(current, u) + outgoing_cost(u)
    # But we should also consider the destination if we're close to the end.
    # When few nodes remain, the last node must go to destination.
    
    # Blend: for the last few nodes, incorporate destination distance
    if n_unvisited <= 5:
        # For each candidate, add a weighted destination term
        dest_dist = distance_matrix[unvisited_nodes, destination_node]
        # Weight increases as we get closer to the end
        weight = 0.5 + 0.5 * (1.0 / n_unvisited)  # higher weight when fewer nodes
        total_cost = dist_from_current + outgoing_cost + weight * dest_dist
    else:
        total_cost = dist_from_current + outgoing_cost
    
    # Select the candidate with minimum total cost
    best_idx = int(np.argmin(total_cost))
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
