"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return destination_node

    if n == 1:
        return int(candidates[0])

    # Pre-compute distance submatrix between candidates: dist_sub[i, j] = d(candidates[i], candidates[j])
    dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    # Distance from current to each candidate
    dist_from_current = distance_matrix[current_node, candidates]

    best_score = float('inf')
    best_node = candidates[0]

    if n == 2:
        # Two nodes c, v. Path: current -> c -> v -> destination
        for i in range(n):
            j = 1 - i
            total = dist_from_current[i] + dist_sub[i, j] + dist_to_dest[j]
            if total < best_score:
                best_score = total
                best_node = candidates[i]
        return int(best_node)

    if n == 3:
        # Three nodes. For candidate c, remaining two v, u.
        # Path: current -> c -> {v or u} -> {other} -> destination
        for i in range(n):
            rem = [k for k in range(n) if k != i]
            a, b = rem[0], rem[1]
            path1 = dist_sub[i, a] + dist_sub[a, b] + dist_to_dest[b]
            path2 = dist_sub[i, b] + dist_sub[b, a] + dist_to_dest[a]
            total = dist_from_current[i] + min(path1, path2)
            if total < best_score:
                best_score = total
                best_node = candidates[i]
        return int(best_node)

    # n > 3: combine 1-step lookahead with destination-awareness.
    # cost_in  = d(current, c)
    # cost_out = min_{v != c} d(c, v)          (nearest unvisited neighbor to leave from)
    # dest_term = w * d(c, destination)         (guide toward eventual closure)
    # The weight w ramps up as few nodes remain so the final stretch aims at the destination.
    # w is small enough to keep the heuristic local yet large enough to prevent
    # disconnected subtours by nudging the tour toward the closing node.
    if n <= 6:
        w = 0.5
    elif n <= 12:
        w = 0.3
    elif n <= 25:
        w = 0.15
    else:
        w = 0.05

    # For each candidate i, compute min distance to any other candidate (excluding self)
    # Use a copy so we can mask the diagonal without disturbing dist_sub.
    for i in range(n):
        # nearest neighbor cost to leave from candidate i
        row = dist_sub[i].copy()
        row[i] = float('inf')
        cost_out = float(np.min(row))

        total = dist_from_current[i] + cost_out + w * dist_to_dest[i]

        if total < best_score:
            best_score = total
            best_node = candidates[i]

    return int(best_node)
# EVOLVE-BLOCK-END
