<<<<<<< SEARCH
def _select_next_lookahead3(current_node, destination_node, unvisited_nodes, distance_matrix):
    k = len(unvisited_nodes)
    U = unvisited_nodes
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, U]
    
    # Get sub-matrix of distances between unvisited nodes
    D = distance_matrix[np.ix_(U, U)]
    
    # Set diagonal to infinity to exclude self
    D_copy = D.copy()
    np.fill_diagonal(D_copy, np.inf)
    
    # Precompute min1 and min2 for each row
    min1_vals = np.full(k, float('inf'))
    min1_idxs = np.full(k, -1, dtype=int)
    min2_vals = np.full(k, float('inf'))
    
    for j in range(k):
        row = D_copy[j]
        # Find two smallest values
        # Using argsort is O(k log k), but k is small (50). 
        # Or just iterate.
        # Let's use argsort for simplicity and speed in numpy.
        indices = np.argsort(row)
        min1_idxs[j] = indices[0]
        min1_vals[j] = row[indices[0]]
        min2_vals[j] = row[indices[1]]
        
    scores = np.full(k, float('inf'))
    
    for i in range(k):
        # We want min over j != i of (D[i, j] + min_next(j, i))
        # min_next(j, i) is min1_vals[j] if min1_idxs[j] != i, else min2_vals[j]
        
        # Create a vector of costs for all j
        # cost_j = D[i, j] + (min1_vals[j] if min1_idxs[j] != i else min2_vals[j])
        
        # Vectorized approach:
        # mask where min1_idxs != i
        mask = (min1_idxs != i)
        next_costs = np.where(mask, min1_vals, min2_vals)
        
        # Exclude j == i (distance is inf in D_copy, but let's be safe)
        # D[i, i] is inf, so it won't be the min.
        total_costs = D[i] + next_costs
        
        # The min over j != i is just the min of total_costs, 
        # because D[i,i] is inf.
        min_lookahead_cost = np.min(total_costs)
        
        scores[i] = dist_from_current[i] + min_lookahead_cost
        
    best_idx = np.argmin(scores)
    return int(U[best_idx])
=======
def _select_next_lookahead4(current_node, destination_node, unvisited_nodes, distance_matrix):
    k = len(unvisited_nodes)
    U = unvisited_nodes
    
    if k == 0:
        return int(current_node)
    if k == 1:
        return int(U[0])
        
    dist_from_current = distance_matrix[current_node, U]
    D = distance_matrix[np.ix_(U, U)]
    
    scores = np.full(k, float('inf'))
    
    for i in range(k):
        # E[j, k_node] = min_{l not in {i, j, k_node}} D[k_node, l]
        E = np.full((k, k), float('inf'))
        
        for k_node in range(k):
            if k_node == i:
                continue
                
            # Get distances from k_node to all nodes
            row = D[k_node].copy()
            
            # Exclude i and k_node
            row[i] = float('inf')
            row[k_node] = float('inf')
            
            # Find min1 and min2
            min1_idx = np.argmin(row)
            min1_val = row[min1_idx]
            
            # Save min1_idx to restore row for min2
            row[min1_idx] = float('inf')
            min2_idx = np.argmin(row)
            min2_val = row[min2_idx]
            
            # Fill column k_node of E
            # For all j, E[j, k_node] is min1_val, except if j == min1_idx, then min2_val
            E[:, k_node] = min1_val
            E[min1_idx, k_node] = min2_val
            
        # C[j, k_node] = D[j, k_node] + E[j, k_node]
        C = D + E
        
        # Mask out i (row and col) and diagonal
        C[i, :] = float('inf')
        C[:, i] = float('inf')
        np.fill_diagonal(C, float('inf'))
        
        # M[j] = min_{k_node != i, j} C[j, k_node]
        # Since C[j, i] is inf and C[j, j] is inf, min over all k_node works.
        M = np.min(C, axis=1)
        
        # Score(i) = dist_from_current[i] + min_{j != i} (D[i, j] + M[j])
        # D[i, j] is row i of D.
        # We need to exclude j=i.
        V = D[i] + M
        V[i] = float('inf')
        
        min_lookahead = np.min(V)
        scores[i] = dist_from_current[i] + min_lookahead
        
    best_idx = np.argmin(scores)
    return int(U[best_idx])
>>>>>>> REPLACE
