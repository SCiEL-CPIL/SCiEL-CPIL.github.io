"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Deadline guard: fall back to nearest-neighbor if time is running out
    if time.time() > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    dm = distance_matrix
    # Cost from current to each candidate
    d_cur = dm[current_node, unv]
    
    # 2-step lookahead: for each candidate, estimate min onward cost to any other unvisited node
    # Build submatrix of distances among unvisited nodes
    sub = dm[np.ix_(unv, unv)]
    # Set diagonal to inf so a node can't pick itself as the onward step
    np.fill_diagonal(sub, np.inf)
    # For each candidate, the min onward distance to any other unvisited
    d_onward = sub.min(axis=1)
    
    # Blended score: immediate cost + weight * onward cost
    # Weight balances short-term vs long-term; 0.5 is a reasonable starting point
    weight = 0.5
    scores = d_cur + weight * d_onward
    
    best_idx = int(np.argmin(scores))
    return int(unv[best_idx])
# EVOLVE-BLOCK-END
