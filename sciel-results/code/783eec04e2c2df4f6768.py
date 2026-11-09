import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: track elapsed time to avoid timeout
    t_start = time.time()
    
    # Convert to numpy arrays if not already for speed
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return int(depot)
        
    # Filter feasible nodes
    node_demands = np.array(demands)[unvisited]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    # Calculate distances from current to feasible nodes
    dists_curr = distance_matrix[current_node, feasible_nodes]
    
    # For each feasible node, compute a savings-based score:
    # score = dist(current, node) + alpha * min_dist(node, other_feasible_nodes)
    # This encourages visiting nodes that are close to the current node AND
    # have nearby other feasible nodes (cluster formation), reducing total route length.
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute pairwise distances among feasible nodes
    # For efficiency, compute the minimum distance from each feasible node to any other feasible node
    # Use distance matrix submatrix
    feasible_idx = feasible_nodes
    sub_matrix = distance_matrix[np.ix_(feasible_idx, feasible_idx)]
    
    # Set diagonal to inf so min doesn't pick self
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Minimum distance from each feasible node to any other feasible node
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # Score: weighted combination of distance from current and proximity to cluster
    alpha = 0.5  # Weight for cluster proximity
    scores = dists_curr + alpha * min_dist_to_other
    
    # Check deadline before final selection
    if time.time() - t_start > 145:
        # Fallback to nearest neighbor if running out of time
        min_idx = np.argmin(dists_curr)
        return int(feasible_nodes[min_idx])
    
    min_idx = np.argmin(scores)
    best_node = feasible_nodes[min_idx]
    
    return int(best_node)
