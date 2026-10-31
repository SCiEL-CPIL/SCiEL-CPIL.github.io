# EVOLVE-BLOCK-START
import numpy as np

def _fast_mst(nodes, distance_matrix):
    """
    Compute MST cost using Prim's algorithm with dense matrix vectorization.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    sub_d = distance_matrix[np.ix_(nodes, nodes)]
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_cost = 0.0
    
    for _ in range(n):
        # Find min dist node not in MST
        masked_min_dist = np.where(in_mst, np.inf, min_dist)
        u = np.argmin(masked_min_dist)
        
        if min_dist[u] == np.inf:
            break
            
        total_cost += min_dist[u]
        in_mst[u] = True
        
        # Update min_dist
        min_dist = np.minimum(min_dist, sub_d[u, :])
        
    return float(total_cost)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-based estimate of the residual tour cost.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Precompute distances and min values
    # 1. Distances from current to all unvisited
    dist_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distances from destination to all unvisited
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # 3. Min distance from each unvisited node to any other unvisited node (excluding itself)
    # sub_d is N x N
    sub_d = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # min_out[i] = min(sub_d[i, :])
    # Since diagonal is 0, this gives the distance to the nearest neighbor.
    min_out = np.min(sub_d, axis=1)
    
    # 4. Min distance from destination to any other unvisited node (excluding the candidate)
    # We need min(dest, V \ {u}) for each u.
    # Sort dist_dest to find global min and second min.
    sorted_indices = np.argsort(dist_dest)
    min_dest_val = dist_dest[sorted_indices[0]]
    min_dest_idx = sorted_indices[0]
    second_min_dest_val = dist_dest[sorted_indices[1]]
    
    min_in_from_dest = np.full(n_unvisited, min_dest_val)
    # For the node that is the global min, the next best is the second min
    min_in_from_dest[min_dest_idx] = second_min_dest_val
    
    best_score = float('inf')
    best_node = -1
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        immediate_dist = dist_current[i]
        
        # Compute MST of remaining nodes
        remaining_indices = [j for j in range(n_unvisited) if j != i]
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            residual_est = 0.0
        else:
            residual_est = _fast_mst(remaining_nodes, distance_matrix)
            # Add connection costs
            # min_dist_cand_rem is min_out[i]
            # min_dist_dest_rem is min_in_from_dest[i]
            residual_est += min_out[i] + min_in_from_dest[i]
            
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
