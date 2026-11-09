"""TSP constructive heuristic: MST-leaf + geometric slack blend."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 150.0 - 3.0  # seconds of wall-clock budget we actually use


def _now():
    return time.time()


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()
    return _DEADLINE - (_now() - _START_TIME)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction using a blend of:
      1. MST-leaf signal: nodes that are leaves in the MST of the remaining
         subgraph (unvisited + destination) are structurally "exposed" and
         should be visited next to avoid isolated subtours.
      2. Geometric slack: triangle-inequality slack as a proxy for turn angle.
      3. Direct distance: nearest-neighbor component.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Time guard: fall back to pure nearest-neighbour if we are close to the limit
    if _time_left() < 0.3:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])

    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a]
        return int(a) if cost_a <= cost_b else int(b)

    # --- Build subgraph: unvisited nodes + destination ---
    # Nodes in subgraph: unvisited_nodes plus destination_node
    sub_nodes = list(unvisited_nodes)
    if destination_node not in sub_nodes:
        sub_nodes.append(destination_node)
    n_sub = len(sub_nodes)
    
    # Build subgraph distance matrix
    sub_idx = {node: i for i, node in enumerate(sub_nodes)}
    sub_dist = np.zeros((n_sub, n_sub), dtype=np.float64)
    for i, ni in enumerate(sub_nodes):
        for j, nj in enumerate(sub_nodes):
            sub_dist[i, j] = distance_matrix[ni, nj]
    
    # --- Compute MST of subgraph using Prim's algorithm ---
    in_mst = np.zeros(n_sub, dtype=bool)
    min_edge = np.full(n_sub, np.inf)
    min_edge[0] = 0.0
    parent = np.full(n_sub, -1, dtype=int)
    
    for _ in range(n_sub):
        # Find unvisited node with smallest min_edge
        unvisited_mask = ~in_mst
        if not np.any(unvisited_mask):
            break
        # Set visited nodes' min_edge to inf for argmin
        masked_min = np.where(unvisited_mask, min_edge, np.inf)
        u = int(np.argmin(masked_min))
        in_mst[u] = True
        # Update neighbors
        for v in range(n_sub):
            if not in_mst[v] and sub_dist[u, v] < min_edge[v]:
                min_edge[v] = sub_dist[u, v]
                parent[v] = u
    
    # --- Identify MST leaves ---
    # A node is a leaf if it has degree 1 in the MST
    degree = np.zeros(n_sub, dtype=int)
    for v in range(1, n_sub):
        if parent[v] >= 0:
            degree[v] += 1
            degree[parent[v]] += 1
    
    # For each unvisited node, compute its MST degree
    mst_degree = np.zeros(n_unvisited, dtype=float)
    for i, node in enumerate(unvisited_nodes):
        if node in sub_idx:
            mst_degree[i] = degree[sub_idx[node]]
    
    # Normalize degree: lower degree (leaf=1) is better. Use 1/(degree+1) as score
    # Leaf (degree 1) -> 1/2, internal (degree 2+) -> lower
    mst_score = 1.0 / (mst_degree + 1.0)
    
    # --- Geometric slack ---
    d_dc = distance_matrix[destination_node, current_node]
    d_cur_cand = distance_matrix[current_node, unvisited_nodes]
    d_dest_cand = distance_matrix[destination_node, unvisited_nodes]
    
    slack = d_dc + d_cur_cand - d_dest_cand
    np.maximum(slack, 0.0, out=slack)
    
    # Normalize slack and distance to [0,1] range for blending
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    max_dist = np.max(d_cur_cand) if np.max(d_cur_cand) > 0 else 1.0
    
    slack_norm = slack / max_slack
    dist_norm = d_cur_cand / max_dist
    
    # --- Blend scores ---
    # We want to minimize: weight_mst * (1 - mst_score) + weight_slack * slack_norm + weight_dist * dist_norm
    # mst_score is higher for leaves, so (1 - mst_score) penalizes non-leaves
    w_mst = 0.3
    w_slack = 0.3
    w_dist = 0.4
    
    scores = w_mst * (1.0 - mst_score) + w_slack * slack_norm + w_dist * dist_norm
    
    # Slight preference for keeping destination for closing the tour
    dest_mask = (unvisited_nodes == destination_node)
    if np.any(dest_mask) and n_unvisited > 8:
        scores[dest_mask] += 0.05 * dist_norm[dest_mask]
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
