"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Filter to feasible nodes (demand <= rest_capacity)
    if len(unvisited_nodes) == 0:
        return 0
    
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # Score each feasible node:
    # - distance from current node
    # - look-ahead: distance from candidate to nearest other unvisited feasible node
    # This encourages clustering to reduce total route length
    
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # For look-ahead, compute minimum distance from each candidate to any other unvisited feasible node
    # (excluding itself)
    n_feasible = len(feasible_nodes)
    if n_feasible > 1:
        # Get submatrix of distances among feasible nodes
        sub_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(sub_matrix, np.inf)
        # Min distance to any other feasible node
        min_dist_to_other = np.min(sub_matrix, axis=1)
    else:
        min_dist_to_other = np.array([0.0])
    
    # Weighted score: primarily distance from current, plus a small penalty for being far from cluster
    # Lower score is better
    alpha = 0.7
    beta = 0.3
    scores = alpha * dists_from_current + beta * min_dist_to_other
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
