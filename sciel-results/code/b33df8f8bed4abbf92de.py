"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour construction.
    Uses a hybrid heuristic:
    - For small remaining sets (k <= 15), uses Dynamic Programming to find the optimal first step.
    - For larger sets, uses a 3-step lookahead greedy heuristic.
    """
    k = len(unvisited_nodes)
    
    if k == 0:
        return int(current_node)

    if k == 1:
        return int(unvisited_nodes[0])

    # Check time deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # Threshold for exact DP solution for the remaining sub-tour's first step
    if k <= 15:
        return _select_next_dp(current_node, destination_node, unvisited_nodes, distance_matrix)
    else:
        return _select_next_lookahead3(current_node, destination_node, unvisited_nodes, distance_matrix)

def _select_next_dp(current_node, destination_node, unvisited_nodes, distance_matrix):
    k = len(unvisited_nodes)
    U = unvisited_nodes
    
    # Local distance matrix for unvisited nodes
    D = distance_matrix[np.ix_(U, U)]
    start_costs = distance_matrix[current_node, U]
    end_costs = distance_matrix[U, destination_node]
    
    best_cost = float('inf')
    best_idx = 0
    
    # Try each node as the first step
    for i in range(k):
        # Cost to go from current to U[i]
        cost_to_u_i = start_costs[i]
        
        # Remaining nodes
        R_indices = [j for j in range(k) if j != i]
        m = len(R_indices)
        
        if m == 0:
            # Just go to destination
            total_cost = cost_to_u_i + end_costs[i]
        else:
            # DP for path from U[i] through R to destination
            # Map R to local indices 0..m-1
            D_R = D[np.ix_(R_indices, R_indices)]
            start_R = D[i, R_indices]
            end_R = end_costs[R_indices]
            
            # DP state: dp[mask][j] = min cost to start at U[i], visit subset 'mask' of R, ending at R[j]
            size = 1 << m
            dp = np.full((size, m), float('inf'))
            
            # Base cases
            for j in range(m):
                dp[1 << j][j] = start_R[j]
                
            # Fill DP table
            for mask in range(1, size):
                # Iterate over ending nodes in this mask
                for j in range(m):
                    if (mask & (1 << j)) == 0:
                        continue
                    current_cost = dp[mask][j]
                    if current_cost == float('inf'):
                        continue
                    
                    # Try to extend to next node
                    for next_j in range(m):
                        if (mask & (1 << next_j)) != 0:
                            continue
                        new_mask = mask | (1 << next_j)
                        new_cost = current_cost + D_R[j, next_j]
                        if new_cost < dp[new_mask][next_j]:
                            dp[new_mask][next_j] = new_cost
            
            # Find min cost to complete the tour
            full_mask = size - 1
            min_path_cost = float('inf')
            for j in range(m):
                if dp[full_mask][j] + end_R[j] < min_path_cost:
                    min_path_cost = dp[full_mask][j] + end_R[j]
            
            total_cost = cost_to_u_i + min_path_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_idx = i
            
    return int(U[best_idx])

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

# EVOLVE-BLOCK-END
