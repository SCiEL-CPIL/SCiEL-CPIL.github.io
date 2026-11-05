"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit using a modified savings heuristic.
    The score for a candidate node j is:
        score(j) = distance(current_node, j) + distance(j, depot)
    This balances the immediate travel cost with the future cost of returning to the depot,
    encouraging routes that do not wander too far from the depot.
    """
    if not unvisited_nodes:
        return depot

    # Filter unvisited nodes to only those that fit in the remaining capacity
    # We use a list comprehension for clarity, but vectorized ops are faster if possible.
    # Given N=50, a simple loop or list comp is fast enough.
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible_nodes:
        return depot

    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Calculate scores for feasible nodes
    # score(j) = d(current, j) + d(j, depot)
    # We can vectorize this using numpy for speed, though N is small.
    nodes_arr = np.array(feasible_nodes)
    
    # Get distances from current_node to all feasible nodes
    d_current_to_nodes = distance_matrix[current_node, nodes_arr]
    
    # Get distances from all feasible nodes to depot
    d_nodes_to_depot = distance_matrix[nodes_arr, depot]
    
    # Total score
    scores = d_current_to_nodes + d_nodes_to_depot
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    return int(nodes_arr[min_idx])
# EVOLVE-BLOCK-END
