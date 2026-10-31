# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def _compute_mst_cost(nodes, distance_matrix):
    """
    Compute the total weight of the Minimum Spanning Tree (MST) of the given nodes
    using the provided distance matrix.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    sub_matrix = distance_matrix[np.ix_(nodes, nodes)]
    
    try:
        mst = minimum_spanning_tree(sub_matrix)
        mst_weight = np.sum(mst.data) if mst.nnz > 0 else 0.0
        return float(mst_weight)
    except Exception:
        return float('inf')


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-based estimate of the residual tour cost.
    
    The residual cost is estimated as the minimum edge weight in the MST of the
    unvisited nodes (excluding the candidate), which represents the cheapest
    connectivity needed to link the remaining cities.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Pre-compute distances from current and destination to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    best_score = float('inf')
    best_node = -1
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = dist_from_current[i]
        
        # Compute MST of remaining unvisited nodes (excluding candidate)
        remaining_indices = [j for j in range(n_unvisited) if j != i]
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            # No remaining nodes, just go to destination
            residual_est = dist_from_dest[i]
        else:
            # Compute MST of remaining nodes
            mst_cost = _compute_mst_cost(remaining_nodes, distance_matrix)
            
            # Use the minimum edge weight in the MST as the residual connectivity estimate
            # This represents the cheapest link needed to maintain connectivity among remaining nodes
            if mst_cost > 0:
                # Extract submatrix to find min edge weight
                sub_matrix = distance_matrix[np.ix_(remaining_nodes, remaining_nodes)]
                # The MST edges are the selected edges; we can approximate min edge as 
                # the smallest distance from candidate to remaining or use MST structure
                # For simplicity and speed, use the minimum distance from candidate to remaining
                # as a proxy for the connectivity cost
                min_connectivity = np.min(distance_matrix[candidate, remaining_nodes])
                # Also consider distance from destination to remaining
                min_dest_connectivity = np.min(dist_from_dest[remaining_indices])
                
                # Residual estimate: MST cost scaled + connectivity costs
                # The directive says to use "minimum edge weight in the MST"
                # Let's compute it properly from the MST
                try:
                    mst_graph = minimum_spanning_tree(sub_matrix)
                    if mst_graph.nnz > 0:
                        min_mst_edge = np.min(mst_graph.data)
                    else:
                        min_mst_edge = 0.0
                except Exception:
                    min_mst_edge = min_connectivity
                
                # Combine: use min_mst_edge as the primary residual estimate
                # plus connection costs to integrate candidate and destination
                residual_est = min_mst_edge + min_connectivity + min_dest_connectivity
            else:
                residual_est = dist_from_dest[i]
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
