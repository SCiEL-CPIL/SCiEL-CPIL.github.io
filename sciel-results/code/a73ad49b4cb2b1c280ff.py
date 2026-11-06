"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

# Cache for DP results to avoid recomputation within a single tour construction
# Key: (current_node, destination_node, frozenset of unvisited node ids)
# Since unvisited_nodes is an array, we convert to a frozenset for hashing
_dp_cache = {}
_start_time = None


def _get_elapsed():
    """Get elapsed time since program start."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_exceeded(deadline=140.0):
    """Check if we're approaching the time limit."""
    return _get_elapsed() > deadline


def _held_karp_dp(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP for exact optimal suffix when unvisited_nodes is small enough.
    Returns the next node to visit.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return None
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Check if we've computed this state before
    key = (current_node, destination_node, tuple(sorted(unvisited_nodes.tolist())))
    if key in _dp_cache:
        return _dp_cache[key]
    
    # Use bitmask DP for up to ~15 nodes
    if n > 15:
        _dp_cache[key] = None
        return None
    
    # Map unvisited nodes to indices 0..n-1
    node_to_idx = {int(node): i for i, node in enumerate(unvisited_nodes)}
    
    # dp[mask][i] = min cost to start from current_node, visit exactly the set of nodes
    # represented by mask (among unvisited), and end at unvisited node i
    # Then add cost from unvisited[i] to destination
    
    INF = float('inf')
    size = 1 << n
    dp = np.full((size, n), INF, dtype=np.float64)
    parent = np.full((size, n), -1, dtype=np.int32)
    
    # Initialize: from current_node to each single unvisited node
    for i in range(n):
        node_i = unvisited_nodes[i]
        dp[1 << i, i] = distance_matrix[current_node, node_i]
    
    # Fill DP table
    for mask in range(1, size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask, i] == INF:
                continue
            # Try adding node j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                node_i = unvisited_nodes[i]
                node_j = unvisited_nodes[j]
                cost = dp[mask, i] + distance_matrix[node_i, node_j]
                if cost < dp[new_mask, j]:
                    dp[new_mask, j] = cost
                    parent[new_mask, j] = i
    
    # Find best final state: must visit all nodes (mask = size-1)
    full_mask = size - 1
    best_cost = INF
    best_last = -1
    for i in range(n):
        total = dp[full_mask, i] + distance_matrix[unvisited_nodes[i], destination_node]
        if total < best_cost:
            best_cost = total
            best_last = i
    
    # Trace back to find the first node after current
    if best_last == -1:
        _dp_cache[key] = None
        return None
    
    # Reconstruct path to find first node
    mask = full_mask
    last = best_last
    while parent[mask, last] != -1:
        mask = mask ^ (1 << last)
        last = parent[mask, last]
    
    next_node = int(unvisited_nodes[last])
    _dp_cache[key] = next_node
    return next_node


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using DP for small remaining sets, greedy for larger sets.
    """
    global _dp_cache
    
    n = len(unvisited_nodes)
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we're running low, fall back to greedy
    if _time_exceeded(135.0):
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Use DP for small number of remaining nodes (up to 15)
    if n <= 15:
        result = _held_karp_dp(current_node, destination_node, unvisited_nodes, distance_matrix)
        if result is not None:
            return result
    
    # For larger sets, use a lookahead heuristic:
    # Score each candidate by immediate edge + min(next-hop to remaining) + edge to destination
    # This encourages choices that don't strand us far from the return city
    
    candidates = unvisited_nodes
    if len(candidates) == 0:
        return int(current_node)
    
    # Compute immediate distances
    immediate = distance_matrix[current_node, candidates]
    
    # For each candidate, estimate cost to complete the tour from there
    # Use: min distance from candidate to any other unvisited + min distance from any unvisited to destination
    # This is a cheap lower-bound-like estimate
    
    # Vectorized computation of min next-hop for each candidate
    # dist_to_others[i] = min distance from candidates[i] to other unvisited nodes
    # dist_to_dest[i] = distance from candidates[i] to destination
    
    # For efficiency with many candidates, use a simplified score
    # score[i] = immediate[i] + alpha * min_dist_to_others[i] + beta * dist_to_dest[i]
    
    # Compute distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # Compute min distance from each candidate to other unvisited nodes
    # This is expensive for large n, so use a sampling or approximation
    if len(candidates) > 20:
        # Use a subset of candidates for the min computation
        sample_size = min(20, len(candidates))
        sample_indices = np.random.choice(len(candidates), sample_size, replace=False)
        sample_nodes = candidates[sample_indices]
        # For each candidate, min distance to sample
        dist_matrix_subset = distance_matrix[candidates[:, None], sample_nodes[None, :]]
        min_dist_to_others = np.min(dist_matrix_subset, axis=1)
    else:
        dist_matrix_full = distance_matrix[candidates[:, None], candidates[None, :]]
        # Set diagonal to inf to exclude self
        np.fill_diagonal(dist_matrix_full, np.inf)
        min_dist_to_others = np.min(dist_matrix_full, axis=1)
    
    # Combine scores with weights
    alpha = 0.3  # weight for next-hop estimate
    beta = 0.2   # weight for destination closure
    
    scores = immediate + alpha * min_dist_to_others + beta * dist_to_dest
    
    best_idx = np.argmin(scores)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
