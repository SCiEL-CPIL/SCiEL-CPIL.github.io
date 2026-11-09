import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = len(local_opt_tour)
    
    # Copy the distance matrix
    penalized_dist = edge_distance.copy()
    
    # 1. Quadratic penalty based on historical usage
    # This encourages finding structurally different tours by heavily penalizing
    # edges that have been used in many previous iterations.
    # alpha controls the strength.
    alpha = 0.05
    # edge_n_used is a matrix of counts.
    # We square it to create a non-linear penalty.
    quadratic_penalty = alpha * (edge_n_used ** 2)
    
    # Add the quadratic penalty to the distance matrix
    penalized_dist += quadratic_penalty
    
    # 2. Strong penalty for edges in the current local optimum tour
    # This ensures the perturbation is effective and forces 2-opt to break
    # at least one edge of the current tour.
    # We use a penalty slightly larger than the typical edge length to ensure
    # these edges are unattractive to the 2-opt optimizer.
    # Estimate typical edge length from the diagonal or mean of the matrix.
    # Using mean is safe.
    mean_dist = np.mean(edge_distance)
    current_tour_penalty = 0.5 * mean_dist
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        penalized_dist[u, v] += current_tour_penalty
        penalized_dist[v, u] += current_tour_penalty
        
    return penalized_dist
