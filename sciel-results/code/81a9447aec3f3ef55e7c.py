import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _estimate_remaining_cost(current, unvisited, destination, dist_matrix):
    """
    Estimate the cost to complete the tour from 'current' visiting all 'unvisited' and returning to 'destination'
    using a Nearest Neighbor heuristic. This provides a tighter 'upper bound' estimate than a simple lower bound.
    """
    if len(unvisited) == 0:
        return dist_matrix[current, destination]
    
    remaining = list(unvisited)
    total_cost = 0.0
    curr = current
    
    while remaining:
        # Find nearest node in remaining
        # Since N is small (max ~50), a simple loop or np.argmin is fine
        dists = dist_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        total_cost += dists[min_idx]
        curr = next_node
        remaining.pop(min_idx)
        
    total_cost += dist_matrix[curr, destination]
    return total_cost

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
    
    # Map unvisited nodes to 0..n-1
    # dp[mask][i] = min cost to start at current, visit set 'mask', and end at node 'i' (index in unvisited)
    # We want to find the best first step.
    
    # Optimization: We only need the first node.
    # Let's compute the cost of each possible first node 'v' by solving the subproblem:
    # Start at 'current', go to 'v', visit all others, end at 'destination'.
    # This is equivalent to finding the min over all permutations of unvisited\{v} of:
    # dist(current, v) + dist(v, p1) + ... + dist(pk, destination)
    
    # Standard Held-Karp finds the min cost for a fixed end node.
    # We can compute the full DP table for the set of unvisited nodes.
    # State: (mask, last_node_index)
    # dp[mask][i] = min cost to visit set 'mask' (subset of unvisited) ending at unvisited[i]
    
    size = 1 << n
    dp = [[float('inf')] * n for _ in range(size)]
    
    # Base case: visit only node i
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    # Fill DP
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == float('inf'):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    
    # Find the best full path
    full_mask = size - 1
    best_total = float('inf')
    best_first_node = -1
    
    # We need to know which node was the *first* one.
    # We can backtrace, or we can just evaluate the cost for each candidate first node 'k'.
    # Cost if first node is k:
    # dist(current, k) + (min cost to visit all others starting from k and ending at destination)
    # The "min cost to visit all others starting from k" is not directly in dp[full_mask][last].
    # dp[full_mask][last] is cost from current to ... to last, visiting all.
    # So, if we fix the first node k, we are looking for:
    # dist(current, k) + min_{perm of rest} (dist(k, p1) + ... + dist(pk, destination))
    
    # Let's just compute the total cost for each possible first node k by running a mini-Held-Karp or using the dp table.
    # Actually, we can just iterate over all possible 'last' nodes in the full_mask state.
    # But we need to identify the first node.
    
    # Alternative: Just compute the min cost to go from current, visit all, end at destination.
    # min_last dp[full_mask][last] + dist_matrix[unvisited[last], destination]
    # This gives the optimal total cost. But which node is first?
    
    # Let's store parent pointers to reconstruct the path.
    parent = [[-1] * n for _ in range(size)]
    
    # Re-run DP with parent tracking
    dp = [[float('inf')] * n for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == float('inf'):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i
                    
    # Find best end node
    best_end = -1
    best_cost = float('inf')
    for i in range(n):
        total = dp[full_mask][i] + dist_matrix[unvisited[i], destination]
        if total < best_cost:
            best_cost = total
            best_end = i
            
    # Backtrace to find the first node
    # Start from (full_mask, best_end)
    cur_mask = full_mask
    cur_node = best_end
    # We need to go back to the state with mask having only 1 bit set.
    while (cur_mask & (cur_mask - 1)) != 0: # while mask has more than 1 bit
        prev_node = parent[cur_mask][cur_node]
        cur_mask = cur_mask & ~(1 << cur_node)
        cur_node = prev_node
        
    # cur_node is the index of the first node in unvisited
    return int(unvisited[cur_node])

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
    
    # Use exact Held-Karp for small subproblems
    if n_unvisited <= 10:
        return _held_karp_next(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    if n_unvisited == 2:
        a = int(unvisited_nodes[0])
        b = int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b

    # Beam Search Strategy for larger problems
    BEAM_WIDTH = 10
    DEPTH = 5
    
    beam = [(0.0, [], current_node)]
    
    for step in range(DEPTH):
        if not beam:
            break
            
        next_beam = []
        
        for cost_so_far, path, last in beam:
            visited_set = set(path)
            
            for node in unvisited_nodes:
                if node in visited_set:
                    continue
                
                new_cost = cost_so_far + distance_matrix[last, node]
                next_beam.append((new_cost, path + [node], node))
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:BEAM_WIDTH]
        
        # Early termination if time is tight
        if time.time() > _DEADLINE - 2.0:
             if beam and len(beam[0][1]) > 0:
                 return int(beam[0][1][0])
             else:
                 dists = distance_matrix[current_node, unvisited_nodes]
                 best_idx = np.argmin(dists)
                 return int(unvisited_nodes[best_idx])

    # After DEPTH steps, select the path with the lowest estimated total cost
    best_total_cost = float('inf')
    best_first_node = -1
    
    for cost_so_far, path, last in beam:
        visited_set = set(path)
        remaining = [n for n in unvisited_nodes if n not in visited_set]
        remaining_arr = np.array(remaining)
        
        # Use NN completion estimate
        est = _estimate_remaining_cost(last, remaining_arr, destination_node, distance_matrix)
        total_est = cost_so_far + est
        
        if total_est < best_total_cost:
            best_total_cost = total_est
            if len(path) > 0:
                best_first_node = int(path[0])
            else:
                best_first_node = int(unvisited_nodes[0])
                
    if best_first_node == -1:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        best_first_node = int(unvisited_nodes[best_idx])
        
    return best_first_node
