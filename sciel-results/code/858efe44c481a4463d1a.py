import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Time guard
    t0 = time.time()
    if t0 > 145.0:
        # Fallback to pure nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # 2-step lookahead
    # For each candidate v in unvisited_nodes:
    #   cost = dist(current, v) + min_{w in unvisited \ {v}} (dist(v, w) + dist(w, destination))
    
    best_node = -1
    best_score = float('inf')
    
    # Vectorized approach might be tricky due to the inner min depending on v.
    # For n=50, a loop is fine.
    
    # Pre-compute distances from destination to all unvisited nodes
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances among unvisited nodes
    # This is a submatrix of distance_matrix
    # unvisited_nodes are indices.
    # dist_matrix_unvisited[i, j] = distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
    
    # To speed up, we can precompute the matrix of distances between unvisited nodes
    # But doing this every step is O(n^2) memory and time. 
    # Since n is small (50), we can just access the original matrix.
    
    for i, v in enumerate(unvisited_nodes):
        # Cost to come to v
        c1 = dist_from_current[i]
        
        # Cost to leave v to the best next node w, then to destination
        # We need min over w in unvisited \ {v} of (dist(v, w) + dist(w, dest))
        
        if n_unvisited == 2:
            # Only one other node w
            j = 1 - i
            w = unvisited_nodes[j]
            c2 = distance_matrix[v, w] + dist_to_dest[j]
        else:
            # Find w that minimizes dist(v, w) + dist(w, dest)
            # We can vectorize this inner part
            # dist_v_to_unvisited = distance_matrix[v, unvisited_nodes]
            # We need to exclude w=v (index i)
            
            dist_v_to_all = distance_matrix[v, unvisited_nodes]
            # Set dist to v itself to infinity so it's not chosen
            dist_v_to_all[i] = np.inf
            
            # combined_cost = dist_v_to_all + dist_to_dest
            # Find min
            min_c2 = np.min(dist_v_to_all + dist_to_dest)
            c2 = min_c2
            
        total_score = c1 + c2
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(v)
            
    return best_node
