"""Rewritten TSP constructive heuristic: pure nearest-neighbor greedy selection."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        c1 = distance_matrix[current_node, n1] + distance_matrix[n2, destination_node]
        c2 = distance_matrix[current_node, n2] + distance_matrix[n1, destination_node]
        return n1 if c1 <= c2 else n2
    # Pure nearest-neighbor: pick the unvisited node closest to current
    best = unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])]
    return int(best)
# EVOLVE-BLOCK-END
