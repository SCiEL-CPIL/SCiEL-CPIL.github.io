"""TSP constructive heuristic: MST-based global structural penalty + Held-Karp suffix."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: total wall-clock budget for all calls combined.
_DEADLINE = 115.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Picks the next city in a TSP tour.
    
    Strategy:
    - If there are few unvisited nodes (<= 12), use Held-Karp DP for exact optimal suffix.
    - Otherwise, use an MST-based global structural penalty heuristic:
      For each candidate, estimate the cost as:
        dist(current, candidate) + MST_cost(remaining + {candidate, destination})
      This captures global structural constraints of the remaining unvisited set,
      avoiding the myopic trap of pure nearest-neighbor lookahead.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, use simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use exact Held-Karp
    if n_unvisited <= 12:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger sets, use MST-based global structural penalty
    return _mst_select(current_node, destination_node, unvisited_nodes, distance_matrix)


def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP to find the optimal next node.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    node_list = unvisited_nodes.astype(int)
    
    best_cost = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        j = node_list[j_idx]
        remaining = np.delete(node_list, j_idx)
        
        if len(remaining) == 0:
            cost = distance_matrix[current_node, j] + distance_matrix[j, destination_node]
        else:
            m = len(remaining)
            
            if m == 1:
                cost = (distance_matrix[current_node, j] + 
                       distance_matrix[j, remaining[0]] + 
                       distance_matrix[remaining[0], destination_node])
            else:
                # Held-Karp for path from j through remaining to destination
                dp = np.full((1 << m, m), np.inf)
                for k in range(m):
                    dp[1 << k, k] = distance_matrix[j, remaining[k]]
                
                for mask in range(1, 1 << m):
                    for k in range(m):
                        if not (mask & (1 << k)):
                            continue
                        if mask == (1 << k):
                            continue
                        prev_mask = mask ^ (1 << k)
                        for l in range(m):
                            if prev_mask & (1 << l):
                                new_cost = dp[prev_mask, l] + distance_matrix[remaining[l], remaining[k]]
                                if new_cost < dp[mask, k]:
                                    dp[mask, k] = new_cost
                
                full_mask = (1 << m) - 1
                min_suffix = np.min(dp[full_mask, :])
                cost = (distance_matrix[current_node, j] + min_suffix + 
                       np.min(dp[full_mask, :]))
                # Actually need the return to destination from the last node
                # Let me fix: the DP gives cost from j through all remaining ending at some node
                # Then we add dist(last_node, destination)
                # But we need to find which ending node minimizes total
                # Redo this properly
                best_end_cost = np.inf
                for end_idx in range(m):
                    end_node = remaining[end_idx]
                    path_cost = dp[full_mask, end_idx]
                    total = path_cost + distance_matrix[end_node, destination_node]
                    if total < best_end_cost:
                        best_end_cost = total
                cost = distance_matrix[current_node, j] + best_end_cost
        
        if cost < best_cost:
            best_cost = cost
            best_next = j
    
    return int(best_next)


def _mst_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    MST-based global structural penalty heuristic.
    
    For each candidate A in unvisited:
      score = dist(current, A) + MST_cost({A} ∪ remaining ∪ {destination})
    
    The MST cost captures the global geometric structure of the remaining nodes.
    Choosing a candidate that leaves a more compact (lower-MST) residual set
    tends to produce shorter tours.
    """
    n = len(unvisited_nodes)
    
    # Precompute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        candidate = unvisited_nodes[j_idx]
        remaining = np.delete(unvisited_nodes, j_idx)
        
        # Build the node set for MST: candidate + remaining + destination
        # That's all unvisited nodes + destination
        mst_nodes = np.concatenate((unvisited_nodes, [destination_node]))
        
        # Compute MST cost on this node set using the distance matrix
        mst_cost = _mst_cost(mst_nodes, distance_matrix)
        
        score = dist_from_current[j_idx] + mst_cost
        
        if score < best_score:
            best_score = score
            best_next = candidate
    
    return int(best_next)


def _mst_cost(nodes, distance_matrix):
    """
    Compute the MST cost for a set of nodes using Prim's algorithm.
    nodes: array of node indices
    distance_matrix: full distance matrix
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Use Prim's algorithm with the distance matrix
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    total_cost = 0.0
    
    for _ in range(n):
        # Find the node with minimum distance not yet in MST
        min_dist[in_mst] = np.inf
        u = np.argmin(min_dist)
        in_mst[u] = True
        total_cost += min_dist[u]
        
        # Update distances
        for v in range(n):
            if not in_mst[v]:
                d = distance_matrix[nodes[u], nodes[v]]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    return total_cost
# EVOLVE-BLOCK-END
