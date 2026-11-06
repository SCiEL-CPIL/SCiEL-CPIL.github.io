"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _mst_weight(unvisited, dist_matrix):
    """Compute MST weight of the subgraph induced by unvisited nodes using Prim's."""
    n = len(unvisited)
    if n <= 1:
        return 0.0
    in_mst = [False] * n
    min_edge = [np.inf] * n
    min_edge[0] = 0.0
    total = 0.0
    for _ in range(n):
        # Find min edge not in MST
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i
        in_mst[u] = True
        total += best
        # Update
        for v in range(n):
            if not in_mst[v]:
                d = dist_matrix[unvisited[u], unvisited[v]]
                if d < min_edge[v]:
                    min_edge[v] = d
    return total

def _held_karp_select(current_node, destination_node, unvisited, dist_matrix):
    """Exact TSP path from current through all unvisited to destination using DP."""
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    # Map unvisited to 0..n-1
    idx_map = {node: i for i, node in enumerate(unvisited)}
    
    # dp[mask][i] = min cost to start at current, visit subset mask, end at unvisited[i]
    # mask is bitmask of visited unvisited nodes
    INF = np.inf
    dp = np.full((1 << n, n), INF)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Initialize: go from current to each unvisited node
    for i in range(n):
        dp[1 << i, i] = dist_matrix[current_node, unvisited[i]]
    
    # Fill DP
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask, i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask, i] + dist_matrix[unvisited[i], unvisited[j]]
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i
    
    # Find best ending: must visit all, then go to destination
    full_mask = (1 << n) - 1
    best_i = 0
    best_cost = INF
    for i in range(n):
        cost = dp[full_mask, i] + dist_matrix[unvisited[i], destination_node]
        if cost < best_cost:
            best_cost = cost
            best_i = i
    
    # The first node is the one with mask = 1 << first_i
    # Trace back to find first node
    mask = full_mask
    i = best_i
    while parent[mask, i] != -1:
        mask ^= (1 << i)
        i = parent[mask, i]
    # Now i is the first node visited
    return int(unvisited[i])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Time guard
    if time.time() - _start_time > _TIME_LIMIT:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    unvisited = np.asarray(unvisited_nodes)
    n = len(unvisited)
    
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    # Use exact DP for small sets
    if n <= 12:
        return _held_karp_select(current_node, destination_node, unvisited, distance_matrix)
    
    # For larger sets: use a heuristic score
    # Score each candidate: 
    #   primary: dist(current, candidate)
    #   secondary: min dist from candidate to other unvisited (connectivity)
    #   tertiary: structural penalty based on MST position
    
    # Precompute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Precompute min distance from each unvisited to any other unvisited
    # For each i, min_j dist(unvisited[i], unvisited[j]) for j != i
    sub_dist = distance_matrix[np.ix_(unvisited, unvisited)]
    np.fill_diagonal(sub_dist, np.inf)
    min_to_other = np.min(sub_dist, axis=1)
    
    # Compute MST weight for structural context
    mst_w = _mst_weight(unvisited, distance_matrix)
    n_unv = len(unvisited)
    avg_mst_edge = mst_w / max(n_unv - 1, 1)
    
    # Score: lower is better
    # Combine: immediate cost + connectivity cost + structural normalization
    # Weight: 0.7 * normalized_immediate + 0.3 * normalized_connectivity
    imm_min = np.min(dist_from_current)
    imm_max = np.max(dist_from_current)
    imm_range = imm_max - imm_min if imm_max > imm_min else 1.0
    
    conn_min = np.min(min_to_other)
    conn_max = np.max(min_to_other)
    conn_range = conn_max - conn_min if conn_max > conn_min else 1.0
    
    norm_imm = (dist_from_current - imm_min) / imm_range
    norm_conn = (min_to_other - conn_min) / conn_range
    
    score = 0.7 * norm_imm + 0.3 * norm_conn
    
    best_idx = np.argmin(score)
    
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
