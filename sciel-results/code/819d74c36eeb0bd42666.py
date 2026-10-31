import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead greedy heuristic.
    
    For each candidate node u, we estimate the cost of the next two moves:
    1. Move from current to u.
    2. Move from u to the nearest unvisited node v (excluding u).
    3. Move from v to the nearest unvisited node w (excluding u, v).
    
    Score(u) = dist(current, u) + min_v dist(u, v) + min_w dist(v, w)
    where v is chosen to minimize the second term, and w to minimize the third.
    
    This approach family (Lookahead) is distinct from the weighted sum heuristic.
    """
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if len(unvisited_nodes) == 2:
        # Simple nearest neighbor for the last two
        d1 = distance_matrix[current_node, unvisited_nodes[0]]
        d2 = distance_matrix[current_node, unvisited_nodes[1]]
        return int(unvisited_nodes[0]) if d1 < d2 else int(unvisited_nodes[1])

    unvisited_idx = unvisited_nodes
    n_unvisited = len(unvisited_idx)
    
    # Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_idx]
    
    # Submatrix of distances between unvisited nodes
    # shape: (n_unvisited, n_unvisited)
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # We need to calculate Score(u) for each u in 0..n_unvisited-1
    # Score(u) = dist_from_current[u] + min_{v != u} (submatrix[u, v] + min_{w != u, v} submatrix[v, w])
    
    # To do this efficiently:
    # 1. For each v, find the min distance to any w != v. Let this be min_dist_v.
    #    Note: This min_dist_v is the best *second* step from v.
    #    However, the 'w' must also be != u.
    #    If the best w for v is u, we need the second best w for v.
    
    # Let's compute for each v:
    # min1_v: smallest dist to any other node
    # min2_v: second smallest dist to any other node
    
    # Set diagonal to inf
    submatrix_copy = submatrix.copy()
    np.fill_diagonal(submatrix_copy, np.inf)
    
    # Get two smallest values for each row
    # argsort is O(N log N), but for N=50 it's fine. Or just sort.
    # sorted_indices = np.argsort(submatrix_copy, axis=1)
    # min1_val = submatrix_copy[np.arange(n_unvisited), sorted_indices[:, 0]]
    # min2_val = submatrix_copy[np.arange(n_unvisited), sorted_indices[:, 1]]
    
    # Since N is small, we can just sort.
    sorted_dists = np.sort(submatrix_copy, axis=1)
    min1_val = sorted_dists[:, 0]
    min2_val = sorted_dists[:, 1]
    
    # We also need to know *which* node is the min1 neighbor for each v, to check if it's u.
    # If min1_neighbor(v) == u, then the best w for v (given u is visited) is min2_val.
    # Otherwise, it's min1_val.
    
    # Get indices of the smallest distances
    # np.argmin returns the first occurrence. Since we set diagonal to inf, it won't be self.
    min1_idx = np.argmin(submatrix_copy, axis=1)
    
    scores = np.zeros(n_unvisited)
    
    for u in range(n_unvisited):
        cost_to_u = dist_from_current[u]
        
        # Find best v for this u
        # v must be != u
        # We want to minimize: submatrix[u, v] + (min_dist_from_v_excluding_u)
        
        # Iterate over potential v? That's O(N) per u, total O(N^2).
        # This is fast enough.
        best_v_cost = np.inf
        for v in range(n_unvisited):
            if v == u:
                continue
            
            cost_u_to_v = submatrix[u, v]
            
            # Determine best w from v, excluding u and v
            # The best w is min1_idx[v] if min1_idx[v] != u, else the second best.
            # We need the value, not just the index, but we have min1_val and min2_val.
            # If min1_idx[v] == u, then the best available w is the one with distance min2_val[v].
            # Else, the best available w is the one with distance min1_val[v].
            
            if min1_idx[v] == u:
                cost_v_to_w = min2_val[v]
            else:
                cost_v_to_w = min1_val[v]
                
            total_path_cost = cost_u_to_v + cost_v_to_w
            if total_path_cost < best_v_cost:
                best_v_cost = total_path_cost
                
        scores[u] = cost_to_u + best_v_cost
        
    best_u_idx = np.argmin(scores)
    return int(unvisited_idx[best_u_idx])
