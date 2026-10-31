import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s

def _time_exceeded():
    return time.time() > _DEADLINE

def _nearest_neighbor_fallback(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Fallback: pick nearest neighbor to current."""
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    dists = distance_matrix[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Potential field constructive heuristic for TSP.
    
    Models tour construction as a dynamic potential field problem where the next
    node is selected by minimizing:
      1. Euclidean distance from current node (attractive component)
      2. Repulsive potential from all other unvisited nodes (inverse distance sum)
    
    The repulsive term prevents clustering and ensures spatial coverage,
    treating the unvisited set as a fluid that resists local saturation.
    """
    if _time_exceeded():
        return _nearest_neighbor_fallback(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        c1, c2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1 if cost1 <= cost2 else c2)
    
    # Distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from destination to all unvisited
    dists_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Submatrix of pairwise distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Repulsive potential: sum of inverse distances to all other unvisited nodes
    # Nodes that are close to many others have high repulsion (they're in a "crowded" area)
    # We want to AVOID picking from crowded areas to ensure spatial coverage
    # But we also want to pick nodes that are close to current (attractive)
    
    # Compute repulsive potential for each candidate
    # repulsive[i] = sum over j != i of 1/d(i,j)
    # Higher repulsive means more crowded, so we add it to the score (penalty)
    repulsive = np.sum(1.0 / sub_matrix, axis=1)
    
    # Normalize repulsive to [0, 1] for stable weighting
    max_repulsive = np.max(repulsive)
    if max_repulsive > 0:
        repulsive_norm = repulsive / max_repulsive
    else:
        repulsive_norm = np.zeros_like(repulsive)
    
    # Distance from destination: nodes close to destination should be saved for later
    # So we subtract a term proportional to distance from destination
    # (nodes far from destination get a smaller penalty, i.e., we're more willing to visit them now)
    
    # Combined score:
    # score = w_attract * dist(current, c) + w_repel * repulsive_norm[c] - w_dest * dist(dest, c)
    
    # The attractive term pulls toward nearby nodes
    # The repulsive term pushes away from crowded regions
    # The destination term encourages saving destination-close nodes for later
    
    w_attract = 1.0
    w_repel = 0.4
    w_dest = 0.25
    
    score = w_attract * dists_from_current + w_repel * repulsive_norm - w_dest * dists_from_dest
    
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
