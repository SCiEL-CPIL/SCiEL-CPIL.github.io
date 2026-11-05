# EVOLVE-BLOCK-START
import numpy as np
import time

# Cached per-instance data so repeated calls are cheap.
_cache = {}


def _init(current_node, destination_node, unvisited_nodes, distance_matrix):
    key = (id(distance_matrix), distance_matrix.shape[0])
    c = _cache.get(key)
    if c is None or c[0] is not distance_matrix:
        c = (distance_matrix, np.ascontiguousarray(distance_matrix, dtype=np.float64))
        _cache[key] = c
    return c[1]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: the evaluator kills after 150 s. If we are near the limit,
    # fall back to the cheap nearest-neighbour choice (no heavy lookahead).
    try:
        if time.time() > 145.0:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[int(np.argmin(d))])
    except Exception:
        pass

    try:
        dm = _init(current_node, destination_node, unvisited_nodes, distance_matrix)
        U = np.asarray(unvisited_nodes)
        n = U.size
        if n == 0:
            return int(current_node)
        if n == 1:
            return int(U[0])

        c = int(current_node)
        dest = int(destination_node)

        # Immediate edge cost current -> candidate.
        edge = dm[c, U]

        if n <= 3:
            # Exact: pick candidate that minimises edge + cost to close to destination.
            score = edge + dm[U, dest]
            return int(U[int(np.argmin(score))])

        # Lookahead penalty: how hard is it to reach the destination after
        # committing to a candidate? Approximated by the minimum distance from
        # the candidate to any other unvisited node (a cheap lower-bound proxy
        # for the remaining path) plus the destination closure cost.
        # This smooths the pure nearest-neighbour objective into a
        # lookahead-aware surrogate without a full solve.
        #
        # Build the penalty via a vectorised min over the candidate set.
        # For each candidate u: pen[u] = min_{v in U, v != u} dm[u, v]
        # We compute the full sub-matrix and take row-mins, masking the diagonal.
        sub = dm[np.ix_(U, U)]
        np.fill_diagonal(sub, np.inf)
        min_next = sub.min(axis=1)  # cost to move on after u
        pen = min_next + dm[U, dest]  # + cost to eventually close to destination

        # Blend immediate and lookahead costs. Immediate edge dominates; the
        # lookahead term breaks ties and avoids dead-end choices that strand
        # the tour far from the destination.
        score = edge + 0.5 * pen

        return int(U[int(np.argmin(score))])
    except Exception:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
# EVOLVE-BLOCK-END
