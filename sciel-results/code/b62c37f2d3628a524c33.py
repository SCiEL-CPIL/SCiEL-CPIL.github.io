"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check time budget
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # For small sets, use exact Held-Karp DP
    if n_unvisited <= 15:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger sets, use 2-step lookahead with destination awareness
    # Score = dist(current, candidate) + w * min(dist(candidate, other) for other in unvisited) + dest_penalty
    # We also consider the destination node to avoid stranding
    
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    
    # 2-step lookahead: for each candidate, find the minimum distance to any other unvisited node
    # This captures the "onward" cost
    # dist_matrix among unvisited nodes
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate i, the min distance to any other node j != i
    # We can compute this by finding the minimum of each row, excluding the diagonal
    np.fill_diagonal(unvisited_dist_matrix, np.inf)
    min_onward = np.min(unvisited_dist_matrix, axis=1)
    
    # Destination awareness: penalize candidates that are far from destination
    # This helps avoid ending up far from the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score: immediate cost + weighted onward cost + destination penalty
    # Weights: immediate=1.0, onward=0.5, dest=0.1
    score = dist_to_candidates + 0.5 * min_onward + 0.1 * dist_to_dest
    
    return int(unvisited_nodes[np.argmin(score)])

def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Exact DP for small unvisited sets (n <= 15)."""
    n = len(unvisited_nodes)
    
    # Map unvisited nodes to indices 0..n-1
    node_map = {node: i for i, node in enumerate(unvisited_nodes)}
    
    # DP state: dp[mask][i] = min cost to visit all nodes in mask, ending at node i
    # mask is a bitmask of which nodes have been visited
    # i is the index of the last visited node (0..n-1)
    
    # We start from current_node, so the first node we visit can be any of the n unvisited nodes
    # dp[1 << i][i] = dist(current_node, unvisited_nodes[i])
    
    full_mask = (1 << n) - 1
    
    # Initialize DP
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    
    for i in range(n):
        dp[1 << i][i] = distance_matrix[current_node, unvisited_nodes[i]]
    
    # Fill DP table
    for mask in range(1, full_mask + 1):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            # Try extending to node j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
    
    # Find the best final node: min over j of dp[full_mask][j] + dist(unvisited_nodes[j], destination_node)
    best_cost = INF
    best_j = 0
    for j in range(n):
        total_cost = dp[full_mask][j] + distance_matrix[unvisited_nodes[j], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_j = j
    
    return int(unvisited_nodes[best_j])
# EVOLVE-BLOCK-END
