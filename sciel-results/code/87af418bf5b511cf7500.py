"""Cluster-based MST-lookahead heuristic for Capacitated Vehicle Routing Problem.

Instead of scoring each candidate with a myopic combination of savings /
nearest-neighbor / demand-fit, we estimate the cost of committing to each
feasible candidate by projecting the remaining route length using a Minimum
Spanning Tree (MST) lower bound on the unvisited nodes (with the candidate
removed). We pick the candidate that minimizes the projected total route
length: distance(current, candidate) + MST(unvisited \ {candidate}) +
distance to depot. This encourages forming spatially coherent clusters and
reduces route fragmentation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def _mst_lower_bound(nodes, dist):
    """
    Compute the MST weight of a set of nodes using Prim's algorithm.
    nodes: array of node indices
    dist: distance matrix
    Returns: MST total weight (float). If len(nodes) <= 1, returns 0.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0

    # Map nodes to 0..n-1 for local indexing
    # Use a set for visited
    visited = [False] * n
    min_dist = [np.inf] * n
    min_dist[0] = 0.0
    total = 0.0

    for _ in range(n):
        # Find unvisited node with min_dist
        u = -1
        best = np.inf
        for i in range(n):
            if not visited[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        total += min_dist[u]

        # Update neighbors
        node_u = nodes[u]
        for v in range(n):
            if not visited[v]:
                d = dist[node_u, nodes[v]]
                if d < min_dist[v]:
                    min_dist[v] = d

    return total


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a cluster-based MST-lookahead heuristic:
    - For each feasible candidate, estimate the remaining route cost using
      an MST lower bound on the remaining unvisited nodes.
    - Pick the candidate that minimizes the projected total route length.
    Returns 0 (depot) if no feasible node can be added or if time limit is reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)

    # 1. Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]

    if len(feasible) == 0:
        return 0

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    # For each feasible candidate, compute projected route cost
    # Projected cost = dist(current, cand) + MST(unvisited \ {cand}) + dist(depot, cand)
    # Note: The MST is a lower bound on the path visiting all remaining nodes.
    # We add dist(depot, cand) to account for closing the route (roughly).
    # Actually, a better projection: dist(current, cand) + MST(unvisited \ {cand})
    # The final return to depot is handled implicitly by the next iteration or
    # we can add dist(depot, last_node) but we don't know the last node.
    # Let's stick to: dist(current, cand) + MST(unvisited \ {cand})
    # This encourages grouping nearby nodes.

    best_cost = np.inf
    best_node = 0

    # Precompute distances from current to feasible
    d_cur_feas = distance_matrix[current_node, feasible]

    # For each candidate, compute MST of the rest
    # To speed up, if the set is large, we might need to be careful.
    # But 50 customers is small.
    for i, cand in enumerate(feasible):
        # Create the set of remaining unvisited nodes (excluding cand)
        # We need to filter unvisited to exclude cand
        # Since unvisited is a numpy array, we can use a mask
        rem_mask = unvisited != cand
        remaining = unvisited[rem_mask]

        if len(remaining) == 0:
            mst_cost = 0.0
        else:
            mst_cost = _mst_lower_bound(remaining, distance_matrix)

        # Projected total additional cost
        # We add the distance to the candidate and the MST of the rest.
        # We also add a small penalty for the distance from the candidate to the depot?
        # No, the MST is a lower bound on the path connecting the remaining nodes.
        # The actual route will be: current -> cand -> ... -> depot.
        # The cost is dist(current, cand) + path(remaining) + dist(last, depot).
        # MST(remaining) <= path(remaining).
        # So projected cost is a lower bound on the remaining route length.
        # We want to minimize this.
        
        proj_cost = d_cur_feas[i] + mst_cost

        if proj_cost < best_cost:
            best_cost = proj_cost
            best_node = cand

    return int(best_node)

# EVOLVE-BLOCK-END
