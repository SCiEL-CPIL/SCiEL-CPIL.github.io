import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before the 150s kill


def _nearest_neighbor_pick(current_node, unvisited_nodes, distance_matrix):
    """Fast fallback: pure nearest neighbor."""
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic:
        score(u) = d(c, u) + min_{v != u} ( d(u, v) + beta * d(v, dest) )
    where beta grows as the tour nears completion, blending nearest-neighbor
    expansion (early) with destination attraction (late).
    """
    # Time limit guard: fall back to cheap nearest neighbor
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return _nearest_neighbor_pick(current_node, unvisited_nodes, distance_matrix)

    unvisited = unvisited_nodes
    n_unvisited = len(unvisited)

    if n_unvisited == 1:
        return int(unvisited[0])

    if n_unvisited == 2:
        # Two left: pick the one that minimizes d(c,u) + d(u,dest)
        # (the other becomes the last node before returning to dest)
        d_cu = distance_matrix[current_node, unvisited]
        d_ud = distance_matrix[unvisited, destination_node]
        return int(unvisited[np.argmin(d_cu + d_ud)])

    # Distances from current to all unvisited
    d_to_unvisited = distance_matrix[current_node, unvisited]

    # Distances from each unvisited to destination
    d_to_dest = distance_matrix[unvisited, destination_node]

    # Submatrix of distances among unvisited nodes
    sub_matrix = distance_matrix[unvisited][:, unvisited].copy()

    # Exclude self-loops (diagonal is 0, which would be a degenerate "next step")
    np.fill_diagonal(sub_matrix, np.inf)

    # Adaptive destination weight:
    # Early in the tour (many unvisited): small beta -> behave like NN with slight lookahead.
    # Late in the tour (few unvisited): large beta -> strongly pull toward destination.
    # Normalize by fraction of tour remaining.
    frac_remaining = n_unvisited / 50.0  # 50-city instances
    # beta ramps from ~0.1 (early) to ~1.5 (late)
    beta = 0.1 + 1.4 * (1.0 - frac_remaining)

    # 2-step lookahead cost:
    # For each candidate u, the best 2-step continuation cost is:
    #   min_v ( d(u, v) + beta * d(v, dest) )
    # Build the matrix M[u, v] = d(u, v) + beta * d(v, dest)
    M = sub_matrix + beta * d_to_dest[None, :]

    min_exit_cost = np.min(M, axis=1)

    # Total score: immediate step + best continuation
    total_score = d_to_unvisited + min_exit_cost

    best_idx = int(np.argmin(total_score))

    return int(unvisited[best_idx])
