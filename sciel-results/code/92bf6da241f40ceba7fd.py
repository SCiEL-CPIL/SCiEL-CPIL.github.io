"""Rewritten program: Nearest Neighbor heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(depot)

    # Filter to capacity-feasible customers.
    feas = unvisited[demands[unvisited] <= rest_capacity]
    if feas.size == 0:
        return int(depot)

    # Nearest Neighbor: Minimize distance from current node to candidate.
    d_cur = distance_matrix[current_node, feas]
    
    # Tie-break: minimize distance to depot, then node id.
    d_dep = distance_matrix[depot, feas]
    
    order = np.lexsort((feas, d_dep, d_cur))
    best = feas[order[0]]
    return int(best)
# EVOLVE-BLOCK-END
