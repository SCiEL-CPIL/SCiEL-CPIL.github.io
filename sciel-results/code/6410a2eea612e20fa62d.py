# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Pick the one that gives shorter total: current->A->dest vs current->B->dest
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    # Distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from destination to all unvisited (for closing the tour)
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # For each candidate, estimate the "future cost" by looking at the 
    # minimum distance from that candidate to other unvisited nodes (2-step lookahead)
    # This penalizes candidates that are isolated from the rest
    # To keep it fast, we sample a subset for the lookahead
    if n_unvisited > 10:
        # Use a subset for speed
        indices = np.random.choice(n_unvisited, size=10, replace=False)
        sample_nodes = unvisited_nodes[indices]
        # For each candidate, min distance to sample (excluding itself)
        # dist_matrix between candidates and sample
        cand_to_sample = distance_matrix[unvisited_nodes[:, None], sample_nodes[None, :]]
        # Set diagonal to inf for candidates that appear in sample
        for i, idx in enumerate(indices):
            pos_in_unvisited = np.where(unvisited_nodes == sample_nodes[i])[0]
            if len(pos_in_unvisited) > 0:
                cand_to_sample[pos_in_unvisited[0], i] = np.inf
        min_future = np.min(cand_to_sample, axis=1)
    else:
        # Full lookahead for small sets
        cand_to_cand = distance_matrix[unvisited_nodes[:, None], unvisited_nodes[None, :]]
        np.fill_diagonal(cand_to_cand, np.inf)
        min_future = np.min(cand_to_cand, axis=1)
    
    # Composite score: primary is distance from current, 
    # plus weighted future cost and weighted distance to destination
    # Tuned weights: emphasize current distance, moderate future, small destination penalty
    score = dist_from_current + 0.3 * min_future + 0.1 * dist_to_dest
    
    return int(unvisited_nodes[np.argmin(score)])
# EVOLVE-BLOCK-END
