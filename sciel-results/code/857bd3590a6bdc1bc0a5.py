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
    Uses the minimum edge weight in the MST of unvisited nodes as a proxy
    for the average step cost in the remaining sub-tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Compute MST of the current set of unvisited nodes
    # Use the minimum edge weight in this MST as an estimate of the 
    # "typical" connection cost for the remaining tour
    mst_cost = _compute_mst_cost(unvisited_nodes, distance_matrix)
    
    # Get the minimum edge weight from the MST
    # We need to recompute to get edge weights, or we can approximate
    # Let's compute the MST edges directly to find the minimum weight
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    try:
        mst = minimum_spanning_tree(sub_matrix)
        if mst.nnz > 0:
            min_mst_edge = float(np.min(mst.data))
        else:
            min_mst_edge = 0.0
    except Exception:
        min_mst_edge = 0.0
    
    # If min_mst_edge is very small or zero, use a small epsilon to avoid division issues
    # Actually, we'll use it as a multiplier for the number of remaining steps
    # The idea: residual cost ≈ min_mst_edge * (n_unvisited - 1)
    # This estimates the total cost to traverse the remaining nodes
    
    best_score = float('inf')
    best_node = -1
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # After visiting candidate, remaining nodes count is n_unvisited - 1
        remaining_count = n_unvisited - 1
        
        # Residual cost estimate using minimum MST edge weight
        if remaining_count == 0:
            # No more nodes to visit, just go to destination
            residual_est = distance_matrix[candidate, destination_node]
        else:
            # Estimate: min_mst_edge * remaining_count + connection to destination
            # The min_mst_edge * remaining_count approximates the cost to visit
            # all remaining nodes in the sub-tour
            # Add the distance from candidate to destination as a final connection cost
            # But we also need to account for the fact that the sub-tour will
            # start near candidate and end near destination
            
            # Use: min_mst_edge * remaining_count + 0.5 * (dist(candidate, dest) + min_dist_to_dest)
            # Simpler: min_mst_edge * remaining_count + dist(candidate, destination) / 2
            # The /2 accounts for the fact that we're estimating a path, not a round trip
            
            min_dist_cand_to_remaining = float('inf')
            if remaining_count > 0:
                remaining_indices = [j for j in range(n_unvisited) if j != i]
                remaining_nodes = unvisited_nodes[remaining_indices]
                if len(remaining_nodes) > 0:
                    min_dist_cand_to_remaining = np.min(distance_matrix[candidate, remaining_nodes])
            
            # Combine: immediate + estimated sub-tour cost + connection to destination
            # sub-tour cost ≈ min_mst_edge * remaining_count
            # Plus we need to get from candidate to the first remaining node
            # And from the last remaining node to destination
            # Approximate these with min distances
            
            residual_est = min_mst_edge * remaining_count
            
            # Add connection costs
            if remaining_count > 0:
                # Distance from candidate to nearest remaining node
                residual_est += min_dist_cand_to_remaining
                # Distance from destination to nearest remaining node
                min_dist_dest_to_remaining = np.min(distance_matrix[destination_node, remaining_nodes])
                residual_est += min_dist_dest_to_remaining
            else:
                residual_est = distance_matrix[candidate, destination_node]
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
