import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    modified = edge_distance.copy()
    
    n = len(local_opt_tour)
    max_dist = np.max(edge_distance)
    big_val = max_dist * 10.0  # Very large value to force edge removal
    
    # For each edge in the current tour, check if there's a beneficial 2-opt swap
    # involving that edge. If yes, set that edge's distance to a very large value
    # to force the subsequent 2-opt to explore the specific alternative connections.
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Find best 2-opt swap that removes edge (u,v)
        best_gain = 0.0
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            a = local_opt_tour[j]
            b = local_opt_tour[(j + 1) % n]
            gain1 = edge_distance[u, v] + edge_distance[a, b] - edge_distance[u, a] - edge_distance[v, b]
            gain2 = edge_distance[u, v] + edge_distance[a, b] - edge_distance[u, b] - edge_distance[v, a]
            gain = max(gain1, gain2)
            if gain > best_gain:
                best_gain = gain
        
        if best_gain > 0:
            # Strong penalty: force 2-opt to remove this edge
            modified[u, v] = big_val
            modified[v, u] = big_val
    
    # Add small penalty for historically used edges to avoid cycling
    if edge_n_used.max() > 0:
        history_penalty = max_dist * 0.05
        modified += history_penalty * edge_n_used
    
    return modified
