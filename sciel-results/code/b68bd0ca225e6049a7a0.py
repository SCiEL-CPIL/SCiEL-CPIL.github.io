# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE = 115.0  # Leave 5s margin for 120s limit

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _reset_state():
    global _start_time
    _start_time = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Reset if this looks like the first call of a new evaluation
    # We don't know for sure, but if elapsed is huge, reset
    if _start_time is not None and _get_elapsed() > 110:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Check deadline
    elapsed = _get_elapsed()
    if elapsed > _DEADLINE:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # Strategy:
    # 1. If n_unvisited is small (<= 12), use Held-Karp DP for exact optimal completion.
    # 2. If n_unvisited is larger, use a greedy completion cost heuristic.
    #    Score(u) = d(cur, u) + Cost_Greedy_Complete(u, U\{u}, dest)
    
    if n_unvisited <= 12:
        # Held-Karp DP
        nodes = unvisited_nodes
        n = len(nodes)
        
        # dp[mask][last] = min cost to visit set mask (bitmask) ending at node 'last' (index in nodes)
        # Start from current_node, which is not in nodes.
        
        size = 1 << n
        dp = np.full((size, n), np.inf)
        
        # Initialize
        for i in range(n):
            dp[1 << i, i] = distance_matrix[current_node, nodes[i]]
        
        # Fill DP
        for mask in range(size):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                cost = dp[mask, i]
                if cost == np.inf:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = cost + distance_matrix[nodes[i], nodes[j]]
                    if new_cost < dp[new_mask, j]:
                        dp[new_mask, j] = new_cost
        
        # Find the best completion
        full_mask = size - 1
        best_cost = np.inf
        best_next = int(nodes[0])
        
        for i in range(n):
            if dp[full_mask, i] < np.inf:
                total_cost = dp[full_mask, i] + distance_matrix[nodes[i], destination_node]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_next = int(nodes[i])
                    
        # Backtrack to find the FIRST node in the optimal path
        mask = full_mask
        last_idx = -1
        for i in range(n):
            if dp[full_mask, i] + distance_matrix[nodes[i], destination_node] == best_cost:
                last_idx = i
                break
        
        if last_idx == -1:
            # Fallback
            return int(nodes[np.argmin(distance_matrix[current_node, nodes])])
            
        path = [last_idx]
        current_mask = full_mask
        curr_idx = last_idx
        
        while (1 << curr_idx) != current_mask:
            prev_mask = current_mask ^ (1 << curr_idx)
            if prev_mask == 0:
                break
            min_cost = np.inf
            prev_idx = -1
            for j in range(n):
                if prev_mask & (1 << j):
                    c = dp[prev_mask, j] + distance_matrix[nodes[j], nodes[curr_idx]]
                    if c < min_cost:
                        min_cost = c
                        prev_idx = j
            if prev_idx == -1:
                break
            path.append(prev_idx)
            curr_idx = prev_idx
            current_mask = prev_mask
            
        first_idx = path[-1]
        return int(nodes[first_idx])

    else:
        # Heuristic for larger n: Greedy Completion Cost
        # Score(u) = d(cur, u) + Greedy_Tour_Cost(start=u, nodes=U\{u}, end=dest)
        
        # Precompute sub-matrix of distances between unvisited nodes
        # unvisited_nodes is an array of indices
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # Distances from current to all unvisited
        dist_from_current = distance_matrix[current_node, unvisited_nodes]
        
        best_score = np.inf
        best_node_idx = 0
        
        n = n_unvisited
        
        for i in range(n):
            # Cost to reach node i from current
            cost = dist_from_current[i]
            
            # Simulate greedy completion from node i
            # Start at i, visit all other unvisited nodes, end at destination
            curr_idx = i
            visited = np.zeros(n, dtype=bool)
            visited[i] = True
            
            for _ in range(n - 1):
                # Find nearest unvisited node to curr_idx
                # We want min sub_dist[curr_idx, j] where not visited[j]
                # Create a masked distance vector
                # To avoid creating a new array every time, we can use a trick or just loop
                # Since n <= 50, a simple loop or np.argmin with mask is fine.
                # Using np.argmin with a temporary mask is O(N)
                
                # Mask out visited nodes
                # We need to find j such that not visited[j] and sub_dist[curr_idx, j] is min
                # Let's use a small loop for speed in Python for N=50? 
                # Or vectorized:
                masked_dists = sub_dist[curr_idx].copy()
                masked_dists[visited] = np.inf
                next_idx = np.argmin(masked_dists)
                
                cost += sub_dist[curr_idx, next_idx]
                visited[next_idx] = True
                curr_idx = next_idx
                
            # Add cost to return to destination
            cost += distance_matrix[unvisited_nodes[curr_idx], destination_node]
            
            if cost < best_score:
                best_score = cost
                best_node_idx = i
                
        return int(unvisited_nodes[best_node_idx])

# EVOLVE-BLOCK-END
