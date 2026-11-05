"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # Ensure adjacency is a numpy array
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Compute degree
    degree = adj.sum(axis=1)

    # Compute neighbor-degree-sum (opportunity cost signal)
    # For each vertex i, sum of degrees of its neighbors
    neighbor_degree_sum = adj @ degree

    # Compute Conflict Radius: number of edges among neighbors
    # This measures local clustering density.
    # A(v) = adjacency matrix
    # A_sq[v, u] = number of common neighbors of v and u
    # For a vertex v, sum of A_sq[v, u] for u in N(v) is 2 * (edges in N(v))
    try:
        if time.time() < deadline:
            A_sq = adj @ adj.T
            # Element-wise product of adj and A_sq, then sum over neighbors
            # adj[v, u] is 1 if u is neighbor of v
            # A_sq[v, u] is common neighbors
            # Sum gives 2 * conflict_radius
            conflict_radius = 0.5 * (adj * A_sq).sum(axis=1)
        else:
            conflict_radius = np.zeros_like(degree)
    except Exception:
        conflict_radius = np.zeros_like(degree)

    # Normalize degree to [0, 1]
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)

    # Normalize conflict radius to [0, 1]
    cr_min = conflict_radius.min()
    cr_max = conflict_radius.max()
    if cr_max > cr_min:
        cr_norm = (conflict_radius - cr_min) / (cr_max - cr_min)
    else:
        cr_norm = np.zeros_like(conflict_radius)

    # Composite score:
    # Directive: prioritize lowest conflict radius.
    # Also low degree is generally beneficial for MIS.
    # We negate both because higher score is selected first.
    w1 = 1.0
    w2 = 0.5

    score = -w1 * degree_norm - w2 * cr_norm

    return score
# EVOLVE-BLOCK-END
