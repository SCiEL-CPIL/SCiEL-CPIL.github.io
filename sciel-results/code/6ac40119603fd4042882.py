import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node for the vehicle to visit.
    Uses a Clarke-Wright Savings based heuristic.
    """
    if not unvisited_nodes:
        return depot

    # Convert to numpy array for efficient vector operations
    unvisited = np.array(unvisited_nodes)
    
    # Filter by capacity
    # demands is an array, unvisited is an array of indices
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If starting from depot, use Nearest Neighbor
    if current_node == depot:
        dists = distance_matrix[depot, feasible_nodes]
        idx = np.argmin(dists)
        return int(feasible_nodes[idx])
    
    # Calculate savings scores
    # Score(j) = d(current, depot) + d(depot, j) - d(current, j)
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    scores = d_current_depot + d_depot_j - d_current_j
    
    # Pick the node with the maximum savings
    idx = np.argmax(scores)
    return int(feasible_nodes[idx])
