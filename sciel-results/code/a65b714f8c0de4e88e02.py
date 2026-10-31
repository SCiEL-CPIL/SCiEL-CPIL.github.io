# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance, destination distance, and a fast approximation of the
    residual tour cost based on the sum of minimum incident edges (a proxy for
    MST connectivity) of the remaining unvisited nodes.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Simple case: pick the one that minimizes the total path cost
        # current -> u -> dest (since the other will be visited next and must return to dest)
        # Actually, if 2 nodes left, we pick u, then the other v is picked.
        # Tour: current -> u -> v -> destination.
        # Cost: dist(current, u) + dist(u, v) + dist(v, dest)
        # We compare u vs v.
        u = unvisited_nodes[0]
        v = unvisited_nodes[1]
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        if cost_u < cost_v:
            return int(u)
        else:
            return int(v)

    # Extract submatrix for unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    n = n_unvisited
    
    # Set diagonal to inf to avoid self-loops
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Compute 1st and 2nd nearest neighbors for each node in the submatrix
    # d1[i]: distance to 1st nearest neighbor
    # nn1[i]: local index of 1st nearest neighbor
    d1 = np.min(sub_matrix, axis=1)
    nn1 = np.argmin(sub_matrix, axis=1)
    
    # Mask the 1st nearest neighbor to find the 2nd
    sub_matrix_masked = sub_matrix.copy()
    sub_matrix_masked[np.arange(n), nn1] = np.inf
    
    d2 = np.min(sub_matrix_masked, axis=1)
    nn2 = np.argmin(sub_matrix_masked, axis=1) # Not strictly needed for the formula, but good to have
    
    # Base residual cost: Sum of minimum incident edges for all nodes
    base_sum = np.sum(d1)
    
    # Precompute dependents: for each node u, which nodes v have u as their 1st nearest neighbor?
    dependents = [[] for _ in range(n)]
    for v in range(n):
        dependents[nn1[v]].append(v)
    
    best_score = float('inf')
    best_node = -1
    
    for u_local in range(n):
        candidate = unvisited_nodes[u_local]
        
        # Immediate cost: current -> candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Destination cost: candidate -> destination
        dest_dist = distance_matrix[candidate, destination_node]
        
        # Residual cost estimate for the remaining nodes (S \ {u})
        # Approximation: Sum of min incident edges for nodes in S \ {u}
        # We start with base_sum (sum for all S) and adjust for removal of u.
        # 1. Remove u's own contribution: - d1[u]
        # 2. For any v where u was the 1st NN, the new min is d2[v] instead of d1[v].
        #    So add (d2[v] - d1[v]) for all v in dependents[u].
        
        residual_est = base_sum - d1[u_local]
        
        for v in dependents[u_local]:
            residual_est += d2[v] - d1[v]
            
        # Scale factor: The sum of min incident edges is typically larger than the MST.
        # A tour has N edges. MST has N-1 edges.
        # Sum of min incident edges is roughly 2 * MST? No, it's an upper bound on MST?
        # Actually, MST <= 0.5 * Sum_Min_Incident? No.
        # Let's use 0.5 as a reasonable scaling factor to balance with the direct distances.
        # The current program used raw MST + min_dists.
        # MST is sum of N-1 edges.
        # Sum_Min_Incident is sum of N edges (each node contributes 1).
        # So they are of similar magnitude.
        # However, Sum_Min_Incident is generally > MST.
        # Let's use 1.0 to be safe, or 0.5.
        # Given the heuristic nature, 1.0 is a strong penalty for "scattered" remaining nodes.
        
        total_score = immediate_dist + dest_dist + 0.5 * residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
