import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Time guard
    if time.time() - _start_time > TIME_LIMIT:
        if not unvisited_nodes:
            return 0
        un_list = list(unvisited_nodes)
        feasible = [u for u in un_list if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda u: distance_matrix[current_node][u]))

    if not isinstance(unvisited_nodes, np.ndarray):
        unvisited_nodes = np.array(unvisited_nodes, dtype=int)
        
    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity
    node_demands = demands[unvisited_nodes]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Get distances from current node to feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get submatrix of distances between feasible nodes
    sub_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Calculate "cluster cohesion" score for each node:
    # Mean distance from node u to all other feasible nodes.
    # This acts as a proxy for distance from the "centroid" of the remaining feasible set.
    # Lower score means the node is more central to the remaining cluster.
    
    # Sum of distances to all other nodes (diagonal is 0, so sum is sum to others)
    # Divide by (N-1) to get mean
    n_feasible = len(feasible_nodes)
    if n_feasible > 1:
        cluster_cohesion = sub_matrix.sum(axis=1) / (n_feasible - 1)
    else:
        cluster_cohesion = np.zeros(n_feasible)
        
    # Heuristic Score:
    # Balance immediate cost (dist_from_current) and future compactness (cluster_cohesion).
    # We want to pick nodes that are close to current AND close to the rest of the cluster.
    # This encourages completing dense clusters.
    
    # Tuning parameter: 
    # If alpha is 0, it's Nearest Neighbor.
    # If alpha is high, it's "Most Central Node".
    # Let's try a moderate value. 
    # Distance scales are typically similar.
    alpha = 0.5
    
    scores = dist_from_current + alpha * cluster_cohesion
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
