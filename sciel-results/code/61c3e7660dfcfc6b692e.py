"""Constructive TSP heuristic: 2-step lookahead nearest neighbour with destination awareness.

MODEL-AND-OPTIMIZE:
    We model the cost contribution of picking candidate c (from current node)
    as a combination of:
        1. Immediate edge cost: dist(current, c)
        2. Lookahead estimate of the next edge (2-step):
           - min distance from c to any other unvisited node (if |unvisited| > 1)
           - dist(c, destination) (if c is the last unvisited node)
        3. Destination closure cost: dist(c, destination)
    
    score(c) = dist(current, c) + beta * next_cost(c) + gamma * dist(c, destination)
    
    The 2-step lookahead avoids locally optimal but globally suboptimal choices
    by penalizing candidates that lead to expensive subsequent edges.
    The destination term helps prevent drift away from the start node.
    Weights are adapted based on fraction of nodes remaining.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150 s wall-clock.
_RUN_START = time.perf_counter()
_DEADLINE_S = 110.0  # safety margin well under the 150 s kill switch


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Safety: if (unexpectedly) no candidates remain, close to the destination.
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)

    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # fall back to the cheapest deterministic choice (pure NN) immediately.
    if (time.perf_counter() - _RUN_START) > _DEADLINE_S:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    n_cities = distance_matrix.shape[0]
    
    # Adaptive weights based on fraction of nodes remaining
    fraction_remaining = n_unvisited / n_cities
    
    # Beta: weight on 2-step lookahead, higher when many nodes remain
    beta = 0.1 + 0.4 * fraction_remaining
    
    # Gamma: weight on destination closure, higher near the end
    gamma = 0.3 * (1.0 - fraction_remaining)

    # Immediate edge costs from current node to each candidate.
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Destination closure costs: distance from each candidate to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]

    # Lookahead cost: for each candidate c, estimate the cost of the next edge.
    if n_unvisited == 1:
        # Last candidate: the next edge goes to destination.
        next_cost = distance_matrix[unvisited_nodes[0], destination_node]
        scores = d_current + beta * next_cost + gamma * d_dest
    else:
        # For each candidate, find the minimum distance to any OTHER unvisited node.
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to inf so we don't pick the node itself as the "next" node.
        np.fill_diagonal(sub_dist, np.inf)
        # For each candidate i, the lookahead is min_j dist(i, j) for j != i.
        min_next = np.min(sub_dist, axis=1)
        scores = d_current + beta * min_next + gamma * d_dest

    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
