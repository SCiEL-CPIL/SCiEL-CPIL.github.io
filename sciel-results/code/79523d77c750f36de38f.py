"""Rewritten program using Savings Heuristic for TSP constructive phase."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 110.0  # 10 second safety margin before 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using the Savings Heuristic.
    
    Args:
        current_node: The node we are currently at.
        destination_node: The node we must eventually return to (start of tour).
        unvisited_nodes: Array of indices of unvisited nodes.
        distance_matrix: NxN matrix of Euclidean distances.
        
    Returns:
        Index of the next node to visit.
    """
    # If no unvisited nodes, return -1 (though contract implies unvisited is non-empty when called)
    if len(unvisited_nodes) == 0:
        return -1

    # Deadline check: if we are running out of time, fall back to nearest neighbor
    # to ensure we return quickly and safely.
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Calculate savings for each candidate node
    # Savings(c, u, d) = dist(c, d) - dist(c, u) - dist(u, d)
    # We want to MAXIMIZE savings.
    
    c = current_node
    d = destination_node
    u_nodes = unvisited_nodes
    
    # Vectorized calculation for efficiency
    # dist_c_d is a scalar
    dist_cd = distance_matrix[c, d]
    
    # dist_c_u is an array of distances from c to each u in u_nodes
    dist_cu = distance_matrix[c, u_nodes]
    
    # dist_u_d is an array of distances from each u in u_nodes to d
    dist_ud = distance_matrix[u_nodes, d]
    
    # Calculate savings
    savings = dist_cd - dist_cu - dist_ud
    
    # To handle potential floating point noise or ties, we can add a tiny penalty for distance
    # or simply pick the max savings. If savings are very close, closer node is generally better
    # for local optimality. Let's add a small epsilon * distance to break ties towards closer nodes.
    # However, pure savings is the standard heuristic. Let's stick to max savings first.
    
    # Find index of max savings
    best_idx_in_unvisited = np.argmax(savings)
    
    # If there are multiple nodes with same max savings (within floating point tolerance),
    # we might want to pick the one with smallest dist_cu to keep the tour tight locally.
    # Let's check for ties.
    max_saving = savings[best_idx_in_unvisited]
    # Tolerance for floating point comparison
    tol = 1e-9
    tied_mask = np.abs(savings - max_saving) < tol
    
    if np.sum(tied_mask) > 1:
        # Among tied savings, pick the one with minimum distance from current node
        tied_indices = np.where(tied_mask)[0]
        tied_dists = dist_cu[tied_indices]
        best_tied_idx = tied_indices[np.argmin(tied_dists)]
        best_idx_in_unvisited = best_tied_idx

    return int(u_nodes[best_idx_in_unvisited])
# EVOLVE-BLOCK-END
