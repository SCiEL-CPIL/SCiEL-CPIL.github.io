"""Improved constructive TSP heuristic: blended 2-lookahead with regret penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _select_blended(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Pick next node using a blend of nearest cost and 2-step lookahead + regret."""
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])

    dm = distance_matrix
    # Cost from current to each candidate
    d_cur = dm[current_node, unv]

    # For 2-step lookahead: min onward distance from each candidate to any other unvisited.
    # Build submatrix of distances among unvisited nodes (n x n), then row-min excluding self.
    sub = dm[np.ix_(unv, unv)]
    # Set diagonal to inf so a node can't pick itself as the onward step.
    np.fill_diagonal(sub, np.inf)
    d_onward = sub.min(axis=1)  # min distance from each candidate to another unvisited

    # Total 2-step cost if we go current -> candidate -> best onward
    two_step = d_cur + d_onward

    # Slight destination-awareness: prefer nodes that also get us closer to destination.
    d_dest = dm[unv, destination_node]
    # Normalize d_dest to [0,1] for a gentle bias (only if more than one unvisited).
    d_dest_range = d_dest.max() - d_dest.min()
    if d_dest_range > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_range
    else:
        d_dest_norm = np.zeros_like(d_dest)

    # Blend: primary signal is 2-step; small weight to nearest and destination closeness.
    # Normalize the two cost vectors to comparable magnitude before blending.
    two_step_span = two_step.max() - two_step.min()
    if two_step_span > 1e-12:
        two_step_norm = (two_step - two_step.min()) / two_step_span
    else:
        two_step_norm = np.zeros_like(two_step)

    d_cur_span = d_cur.max() - d_cur.min()
    if d_cur_span > 1e-12:
        d_cur_norm = (d_cur - d_cur.min()) / d_cur_span
    else:
        d_cur_norm = np.zeros_like(d_cur)

    # Weighted score: lower is better.
    score = (0.70 * two_step_norm + 0.20 * d_cur_norm + 0.10 * d_dest_norm)

    idx = int(np.argmin(score))
    return int(unv[idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    return _select_blended(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
