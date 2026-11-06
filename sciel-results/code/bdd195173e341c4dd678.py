import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _mst_cost(nodes, dist_matrix):
    """Compute MST cost using Prim's algorithm on the given nodes."""
    if len(nodes) <= 1:
        return 0.0
    n = len(nodes)
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total = 0.0
    
    for _ in range(n):
        # Find node with minimum distance not in MST
        u = int(np.argmin(min_dist))
        in_mst[u] = True
        total += min_dist[u]
        
        # Update distances
        for v in range(n):
            if not in_mst[v]:
                d = dist_matrix[nodes[u], nodes[v]]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    return total

def _insertion_estimate(current, candidate, unvisited, destination, dist_matrix):
    """
    Estimate tour completion cost if we go from current to candidate next.
    Uses nearest-neighbor insertion to estimate remaining cost.
    """
    if len(unvisited) == 0:
        return dist_matrix[current, destination]
    
    # Remove candidate from unvisited for the remaining estimation
    mask = unvisited != candidate
    remaining = unvisited[mask]
    
    if len(remaining) == 0:
        return dist_matrix[current, candidate] + dist_matrix[candidate, destination]
    
    # Estimate: cost from candidate to visit all remaining and return to destination
    # Use a combination of MST lower bound and nearest-neighbor upper bound
    # For speed, use MST * 1.0 as a fast estimate (MST is a lower bound for TSP)
    
    # Fast approach: MST of remaining nodes + best connections
    # MST gives a lower bound; we add the cost to connect candidate and destination
    
    mst_rem = _mst_cost(remaining, dist_matrix)
    
    # Cost to connect candidate to the remaining set and destination
    # Find nearest remaining node to candidate
    if len(remaining) > 0:
        dists_to_rem = dist_matrix[candidate, remaining]
        min_to_rem = np.min(dists_to_rem)
        
        # Find nearest remaining node to destination
        dists_from_dest = dist_matrix[destination, remaining]
        min_from_dest = np.min(dists_from_dest)
        
        # Estimate: MST + connection costs (approximate)
        # The actual path needs to enter remaining from candidate and exit to destination
        # A reasonable estimate: MST + min_to_rem + min_from_dest
        return dist_matrix[current, candidate] + mst_rem + min_to_rem + min_from_dest
    else:
        return dist_matrix[current, candidate] + dist_matrix[candidate, destination]

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check
    if time.time() > _DEADLINE - 1.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # For small n, use exact Held-Karp
    if n_unvisited <= 10:
        return _held_karp_next(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    # For larger n, use MST-based insertion heuristic
    # Score each candidate by estimated total tour completion cost
    
    # Compute scores for all candidates
    scores = np.empty(n_unvisited, dtype=np.float64)
    for i, node in enumerate(unvisited_nodes):
        scores[i] = _insertion_estimate(current_node, node, unvisited_nodes, destination_node, distance_matrix)
    
    # Select the candidate with minimum estimated cost
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

def _held_karp_next(current, unvisited, destination, dist_matrix):
    """
    Exact solution for the next node when the number of unvisited nodes is small.
    Uses Held-Karp DP to find the shortest path from 'current' visiting all 'unvisited' and ending at 'destination'.
    Returns the first node in that optimal path.
    """
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    size = 1 << n
    INF = float('inf')
    dp = np.full((size, n), INF, dtype=np.float64)
    parent = np.full((size, n), -1, dtype=np.int64)
    
    # Base case: visit only node i
    for i in range(n):
        dp[1 << i, i] = dist_matrix[current, unvisited[i]]
        
    # Fill DP
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask, i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask, i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask, j]:
                    dp[new_mask, j] = cost
                    parent[new_mask, j] = i
                    
    # Find best end node
    full_mask = size - 1
    best_end = -1
    best_cost = INF
    for i in range(n):
        total = dp[full_mask, i] + dist_matrix[unvisited[i], destination]
        if total < best_cost:
            best_cost = total
            best_end = i
            
    # Backtrace to find the first node
    cur_mask = full_mask
    cur_node = best_end
    while (cur_mask & (cur_mask - 1)) != 0:
        prev_node = parent[cur_mask, cur_node]
        cur_mask = cur_mask & ~(1 << cur_node)
        cur_node = prev_node
        
    return int(unvisited[cur_node])
