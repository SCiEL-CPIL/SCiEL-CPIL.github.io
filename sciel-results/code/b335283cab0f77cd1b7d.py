   <<<<<<< SEARCH
    dist_c_j = distance_matrix[current_node, feasible]
    dist_dep_j = distance_matrix[depot, feasible]
    
    scores = dist_c_j - dist_dep_j
    
    # Pick the minimum score (max savings)
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
   =======
    if current_node == depot:
        dists = distance_matrix[depot, feasible]
        best_idx = np.argmin(dists)
        return int(feasible[best_idx])
        
    dist_c_j = distance_matrix[current_node, feasible]
    dist_dep_j = distance_matrix[depot, feasible]
    dist_c_dep = distance_matrix[current_node, depot]
    
    scores = dist_c_j - dist_dep_j
    min_score = np.min(scores)
    
    if min_score >= dist_c_dep:
        return depot
        
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
   >>>>>>> REPLACE
   