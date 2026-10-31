# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic based on the minimum
    edge weight in the MST of the unvisited nodes to estimate residual cost.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Get the submatrix of distances for unvisited nodes
    # This is O(N^2)
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    n = n_unvisited
    
    # Find the global minimum edge in the subgraph
    # We need to exclude the diagonal (self-loops)
    np.fill_diagonal(sub_dist, np.inf)
    
    # Find the index of the minimum value
    min_val_idx = np.unravel_index(np.argmin(sub_dist), sub_dist.shape)
    min_edge_weight = sub_dist[min_val_idx]
    u1 = min_val_idx[0]
    u2 = min_val_idx[1]
    
    # Map local indices to global node IDs
    node_u1 = int(unvisited_nodes[u1])
    node_u2 = int(unvisited_nodes[u2])
    
    # Precompute residual factor for each candidate
    # For most candidates, the min edge in the remaining set is the global min edge.
    # Only for candidates that are part of the global min edge, we need to find the next min edge.
    
    residual_factors = np.full(n, min_edge_weight)
    
    # For u1 and u2, we need to find the min edge in S \ {u}
    # We can do this by masking the row/col and finding min
    
    # Helper to find min edge excluding a specific local index
    def get_min_edge_excluding(excl_idx):
        mask = np.ones((n, n), dtype=bool)
        mask[excl_idx, :] = False
        mask[:, excl_idx] = False
        # Mask diagonal too, though it's inf
        np.fill_diagonal(mask, False)
        masked_dist = np.where(mask, sub_dist, np.inf)
        return np.min(masked_dist)

    # Calculate for u1
    if u1 != u2: # Always true for distinct nodes
        min_excl_u1 = get_min_edge_excluding(u1)
        residual_factors[u1] = min_excl_u1
        
        # Calculate for u2
        min_excl_u2 = get_min_edge_excluding(u2)
        residual_factors[u2] = min_excl_u2
    else:
        # Should not happen
        pass

    # Now select the best node
    best_score = float('inf')
    best_node = -1
    
    # n_unvisited is the current count. After picking one, n-1 remain.
    n_remaining = n_unvisited - 1
    
    for i in range(n_unvisited):
        candidate = int(unvisited_nodes[i])
        
        # Immediate distance
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Residual estimate: 
        # Using "minimum edge weight in MST" as proxy for average edge cost.
        # Total residual cost ~ (number of remaining edges) * (avg edge cost)
        # We use min edge as a lower bound for avg edge cost.
        # The number of edges in a tour visiting k nodes is k.
        # So residual ~ n_remaining * min_edge_weight_excl
        residual_est = n_remaining * residual_factors[i]
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
