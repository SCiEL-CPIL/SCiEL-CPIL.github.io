"""Rewritten program: local-density constructive heuristic with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 110.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # If time is running out, fall back to simple nearest neighbor for speed
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Compute distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]

    # Compute distances from destination node to all unvisited nodes
    dist_to_destination = distance_matrix[destination_node, unvisited_nodes]

    # Local density metric: for each candidate, estimate the "gap" it would create.
    # We score each candidate by a combination of:
    # 1. Distance from current to candidate (immediate step cost)
    # 2. Distance from candidate to destination (future connection cost)
    # 3. A local density penalty: how close the candidate is to other unvisited nodes
    #    (preferring nodes that are not too isolated, to avoid long detours)
    
    # The "nearest-neighbor gap" idea: when we insert candidate i into the partial tour,
    # the incremental cost is roughly dist(current, i) + dist(i, next) - dist(current, next).
    # Since we don't know 'next', we approximate using the destination as a proxy for the
    # eventual continuation, and also consider the local neighborhood.
    
    # Simple but effective: score = alpha * dist_from_current + beta * dist_to_destination
    # with a small penalty for candidates that are far from the local cluster.
    
    # Compute the mean distance from each candidate to all other unvisited nodes as a
    # proxy for "how central" it is. Central nodes tend to lead to shorter tours.
    # This is O(n^2) per call, but n <= 50, so it's fast.
    
    if n_unvisited <= 50:
        # Compute pairwise distances among unvisited nodes
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Mean distance from each node to all others (excluding self)
        mean_dist_to_others = np.mean(sub_matrix, axis=1)
        # Normalize to [0, 1] range for weighting
        min_md = np.min(mean_dist_to_others)
        max_md = np.max(mean_dist_to_others)
        if max_md > min_md:
            normalized_density = (mean_dist_to_others - min_md) / (max_md - min_md)
        else:
            normalized_density = np.zeros(n_unvisited)
    else:
        normalized_density = np.zeros(n_unvisited)

    # Weighted score: lower is better
    # We want to minimize immediate cost and future cost, but also avoid overly
    # isolated nodes (high normalized_density means more isolated)
    alpha = 0.6  # weight for immediate distance
    beta = 0.3   # weight for distance to destination
    gamma = 0.1  # weight for local density penalty

    scores = alpha * dist_from_current + beta * dist_to_destination + gamma * normalized_density

    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)

    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
