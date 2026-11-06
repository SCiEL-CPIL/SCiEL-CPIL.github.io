"""Constructive TSP heuristic with 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 138.0


def _nearest_neighbor(current_node, unvisited, distance_matrix):
    return int(unvisited[np.argmin(distance_matrix[current_node, unvisited])])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited = np.asarray(unvisited_nodes)
    m = unvisited.size
    if m == 0:
        return int(current_node)
    if m == 1:
        return int(unvisited[0])

    # Explicit deadline guard: fall back to cheap, valid choice near the limit.
    if time.monotonic() > _DEADLINE:
        return _nearest_neighbor(current_node, unvisited, distance_matrix)

    # 2-step lookahead: penalize candidates whose best subsequent hop is far.
    sub = distance_matrix[np.ix_(unvisited, unvisited)]
    sub = sub.copy()
    np.fill_diagonal(sub, np.inf)
    row_min = sub.min(axis=1)
    d1 = distance_matrix[current_node, unvisited]
    score = d1 + row_min
    best_idx = int(np.argmin(score))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
