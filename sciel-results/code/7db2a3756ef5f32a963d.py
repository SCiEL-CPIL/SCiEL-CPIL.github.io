# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    best_score = float('inf')
    best_node = -1

    # Pre-compute distances from current and destination to all unvisited
    # dist_to_current = distance_matrix[current_node, unvisited_nodes]
    # dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Residual cost estimation
        remaining_indices = np.array([j for j in range(n_unvisited) if j != i])
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            residual_est = 0.0
        elif len(remaining_nodes) == 1:
            residual_est = distance_matrix[candidate, remaining_nodes[0]] + distance_matrix[remaining_nodes[0], destination_node]
        else:
            # Compute MST of remaining nodes
            sub_matrix = distance_matrix[np.ix_(remaining_nodes, remaining_nodes)]
            try:
                mst = minimum_spanning_tree(sub_matrix)
                if mst.nnz == 0:
                    residual_est = 0.0
                else:
                    # Get the weights of the MST edges
                    edge_weights = mst.data
                    # Sort edge weights to find the smallest ones
                    # The "tightness" of the cluster is indicated by the smallest edges.
                    # We use the sum of the smallest half of the edges as a proxy for the core cost.
                    k = max(1, len(edge_weights) // 2)
                    smallest_edges = np.sort(edge_weights)[:k]
                    core_cost = np.sum(smallest_edges)
                    
                    # Also add the cost to connect candidate and destination to the cluster
                    # Use min distance to remaining nodes
                    min_dist_cand = np.min(distance_matrix[candidate, remaining_nodes])
                    min_dist_dest = np.min(distance_matrix[destination_node, remaining_nodes])
                    
                    residual_est = core_cost + min_dist_cand + min_dist_dest
            except Exception:
                residual_est = float('inf')

        # Combine immediate and residual
        # Weight the residual heavily? Or lightly?
        # Let's try 1.0 for immediate and 1.0 for residual components.
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
