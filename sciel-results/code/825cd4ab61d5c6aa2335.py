"""Rewritten program: TSP constructive heuristic using Clarke-Wright savings principle."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 3 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Clarke-Wright savings-based heuristic.
    
    The "savings" of visiting node c next (instead of going current->dest and dest->c)
    is: d(current, dest) + d(dest, c) - d(current, c)
    
    We pick the candidate that maximizes savings (equivalently minimizes negative savings),
    with a tie-break toward shorter immediate distance.
    
    Additionally, we incorporate a soft penalty for candidates that are far from all
    remaining unvisited nodes (to avoid creating long isolated legs later).
    """
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline early
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Compute distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited nodes to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Savings: how much we save by going current->cand directly vs current->dest->cand
    # If savings is positive, it's better to visit cand before going to dest
    d_current_to_dest = distance_matrix[current_node, destination_node]
    savings = d_current_to_dest + dists_to_dest - dists_from_current
    
    # For isolation penalty: for each candidate, compute mean distance to other unvisited nodes
    # This discourages picking a node that's far from the rest of the cluster
    dists_between = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(dists_between, np.inf)
    
    # Mean distance to other unvisited nodes (penalize isolated points)
    mean_dist_to_others = np.mean(dists_between, axis=1)
    
    # Combine: maximize savings, but penalize isolation
    # Normalize savings and mean_dist to comparable scales
    savings_range = np.max(savings) - np.min(savings)
    if savings_range > 1e-12:
        savings_norm = (savings - np.min(savings)) / savings_range
    else:
        savings_norm = np.zeros(n)
    
    mean_range = np.max(mean_dist_to_others) - np.min(mean_dist_to_others)
    if mean_range > 1e-12:
        mean_norm = (mean_dist_to_others - np.min(mean_dist_to_others)) / mean_range
    else:
        mean_norm = np.zeros(n)
    
    # Score: we want to maximize savings and minimize isolation
    # Use a weighted combination; primary signal is savings
    gamma = 0.15  # Weight for isolation penalty
    scores = -savings_norm + gamma * mean_norm
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
