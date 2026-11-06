# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    # For small n, use Held-Karp exact DP
    if n_unvisited <= 12:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger n, use greedy + 2-opt for each candidate
    best_cost = np.inf
    best_node = unvisited_nodes[0]
    
    for cand in unvisited_nodes:
        # Build greedy path from cand through remaining
        remaining = np.delete(unvisited_nodes, np.where(unvisited_nodes == cand)[0])
        curr = cand
        path = [cand]
        total_cost = distance_matrix[current_node, cand]
        
        while len(remaining) > 0:
            dists = distance_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            total_cost += dists[min_idx]
            remaining = np.delete(remaining, min_idx)
            curr = next_node
            path.append(curr)
            if total_cost >= best_cost:
                break
        
        if len(remaining) == 0:
            total_cost += distance_matrix[curr, destination_node]
            # Apply 2-opt improvement
            improved_cost = _two_opt_improve(path, distance_matrix, current_node, destination_node)
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_node = cand
    
    return int(best_node)

def _two_opt_improve(path, distance_matrix, start_node, end_node):
    """Apply 2-opt with O(1) cost-delta. Returns improved total cost."""
    n = len(path)
    if n < 2:
        return distance_matrix[start_node, path[0]] + distance_matrix[path[0], end_node]
    
    def total_cost(p):
        c = distance_matrix[start_node, p[0]]
        for i in range(len(p) - 1):
            c += distance_matrix[p[i], p[i + 1]]
        c += distance_matrix[p[-1], end_node]
        return c
    
    current_cost = total_cost(path)
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Compute delta for reversing path[i+1..j]
                if j < n - 1:
                    delta = (distance_matrix[path[i], path[j]] + 
                             distance_matrix[path[i+1], path[j+1]] - 
                             distance_matrix[path[i], path[i+1]] - 
                             distance_matrix[path[j], path[j+1]])
                else:
                    delta = (distance_matrix[path[i], path[j]] + 
                             distance_matrix[path[i+1], end_node] - 
                             distance_matrix[path[i], path[i+1]] - 
                             distance_matrix[path[j], end_node])
                
                if delta < -1e-10:
                    # Perform reversal
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    current_cost += delta
                    improved = True
                    break
            if improved:
                break
    
    return current_cost

def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Use Held-Karp DP to find the optimal next node."""
    n = len(unvisited_nodes)
    # Map unvisited nodes to 0..n-1
    node_to_idx = {node: i for i, node in enumerate(unvisited_nodes)}
    
    # Build sub-distance matrix
    sub_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub_dist[i, j] = distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
    
    # dp[S][j] = min cost to visit set S, ending at j, starting from current_node
    # S is a bitmask, j is 0..n-1
    full_mask = (1 << n) - 1
    dp = [[np.inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    # Initialize
    for j in range(n):
        dp[1 << j][j] = distance_matrix[current_node, unvisited_nodes[j]]
    
    # Fill DP
    for S in range(1, 1 << n):
        for j in range(n):
            if not (S & (1 << j)):
                continue
            if dp[S][j] == np.inf:
                continue
            # Try to extend to k not in S
            for k in range(n):
                if S & (1 << k):
                    continue
                new_S = S | (1 << k)
                new_cost = dp[S][j] + sub_dist[j, k]
                if new_cost < dp[new_S][k]:
                    dp[new_S][k] = new_cost
                    parent[new_S][k] = j
    
    # Find best last node
    best_total = np.inf
    best_last = -1
    for j in range(n):
        total = dp[full_mask][j] + distance_matrix[unvisited_nodes[j], destination_node]
        if total < best_total:
            best_total = total
            best_last = j
    
    # Backtrack to find first node
    S = full_mask
    j = best_last
    while parent[S][j] != -1:
        prev = parent[S][j]
        S = S & ~(1 << j)
        j = prev
    
    # j is now the first node
    return int(unvisited_nodes[j])
# EVOLVE-BLOCK-END
