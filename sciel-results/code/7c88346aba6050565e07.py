# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 145.0  # 5s margin

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def _mst_weight(nodes, distance_matrix):
    """
    Calculate the weight of the Minimum Spanning Tree for the given set of nodes.
    Optimized Prim's algorithm.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Map global node ids to local indices 0..n-1
    # We assume nodes is an array of global indices.
    
    # Initialize distances to infinity
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    visited = np.zeros(n, dtype=bool)
    
    total_weight = 0.0
    
    # Pre-extract the sub-matrix for speed? 
    # d_sub = distance_matrix[np.ix_(nodes, nodes)]
    # This is O(N^2) memory and time per call, which is expensive if called N times.
    # Instead, access distance_matrix directly.
    
    for _ in range(n):
        # Find the unvisited node with the smallest min_dist
        # Using argmin on masked array
        masked_min_dist = np.where(visited, np.inf, min_dist)
        u_idx = np.argmin(masked_min_dist)
        
        if masked_min_dist[u_idx] == np.inf:
            break # Disconnected (should not happen)
            
        total_weight += masked_min_dist[u_idx]
        visited[u_idx] = True
        
        # Update distances for unvisited neighbors
        u_global = nodes[u_idx]
        # Get distances from u_global to all nodes
        # We only care about unvisited nodes
        # d_to_all = distance_matrix[u_global, :]
        # We need to update min_dist[v] for v in unvisited
        # This is a vectorized operation
        
        # Get indices of unvisited nodes
        unvisited_mask = ~visited
        if not np.any(unvisited_mask):
            continue
            
        # Get the global indices of the unvisited nodes to index into distance_matrix
        # Actually, we can just compute distances to all nodes and mask the result
        d_to_all = distance_matrix[u_global, :]
        
        # Update min_dist for unvisited nodes
        # min_dist[unvisited_mask] = np.minimum(min_dist[unvisited_mask], d_to_all[nodes[unvisited_mask]])
        
        # To avoid creating a temporary array for nodes[unvisited_mask], we can do:
        # But d_to_all is indexed by global node id.
        # min_dist is indexed by local index.
        # So we need d_to_all[nodes[unvisited_mask]]
        
        new_dists = d_to_all[nodes[unvisited_mask]]
        current_dists = min_dist[unvisited_mask]
        min_dist[unvisited_mask] = np.minimum(current_dists, new_dists)
        
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using an MST-inclusive heuristic.
    Score(u) = d(current, u) + MST((U \ {u}) U {destination})
    """
    if _is_timeout():
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)

    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    # Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # We need to calculate MST for (U \ {u}) U {destination}
    # This set has size n_unvisited - 1 + 1 = n_unvisited
    
    for i, u in enumerate(unvisited_nodes):
        # Immediate cost
        immediate_cost = dist_from_current[i]
        
        # Construct the set of nodes for MST: all unvisited except u, plus destination
        # Note: If destination is in unvisited_nodes, we must be careful not to duplicate it.
        # In standard TSP, destination is the start node, which is already visited.
        # So destination_node is NOT in unvisited_nodes.
        
        remaining_nodes = np.delete(unvisited_nodes, i)
        
        # Add destination to the list
        mst_nodes = np.append(remaining_nodes, destination_node)
        
        # Calculate MST weight
        mst_w = _mst_weight(mst_nodes, distance_matrix)
        
        total_score = immediate_cost + mst_w
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(u)
            
    return best_node

# EVOLVE-BLOCK-END
