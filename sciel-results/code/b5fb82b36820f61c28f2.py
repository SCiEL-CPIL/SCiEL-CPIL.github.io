import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Convert to numpy arrays if not already for speed
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return int(depot)
        
    # Filter feasible nodes
    # demands is likely a list or array indexed by node id
    # Ensure demands is accessible
    node_demands = np.array(demands)[unvisited]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    # Calculate distances from current to feasible nodes
    dists_curr = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances from depot to feasible nodes
    dists_depot = distance_matrix[depot, feasible_nodes]
    
    # Calculate savings: dist(depot, node) - dist(current, node)
    savings = dists_depot - dists_curr
    
    # Find nodes with positive savings
    positive_mask = savings > 0
    positive_nodes = feasible_nodes[positive_mask]
    positive_dists = dists_curr[positive_mask]
    
    if len(positive_nodes) == 0:
        # No node offers savings over returning to depot
        # Return to depot
        return int(depot)
        
    # Among positive savings nodes, pick the one with minimum distance to current
    # This ensures we stay local while capturing savings
    min_idx = np.argmin(positive_dists)
    best_node = positive_nodes[min_idx]
    
    return int(best_node)
