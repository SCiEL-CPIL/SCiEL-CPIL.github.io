import numpy as np
import time

_start_time = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 148.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 2.0:
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    if n == 2:
        v1, v2 = unvisited_nodes
        cost1 = distance_matrix[current_node, v1] + distance_matrix[v1, v2] + distance_matrix[v2, destination_node]
        cost2 = distance_matrix[current_node, v2] + distance_matrix[v2, v1] + distance_matrix[v1, destination_node]
        return int(v1 if cost1 <= cost2 else v2)

    # For n <= 8, do exact optimal completion via DP (Held-Karp)
    if n <= 8:
        # Map unvisited nodes to indices 0..n-1
        node_to_idx = {v: i for i, v in enumerate(unvisited_nodes)}
        
        # dp[mask][i] = min cost to visit all nodes in mask, ending at node i (i in mask)
        # We need to start from current_node, visit all nodes in mask, then go to destination_node
        # Actually, let's define: dp[mask][i] = min cost of a path starting at current_node,
        # visiting exactly the set of nodes in mask, and ending at unvisited_nodes[i]
        
        full_mask = (1 << n) - 1
        dp = np.full((1 << n, n), float('inf'))
        
        # Base case: visit just node i
        for i in range(n):
            dp[1 << i, i] = distance_matrix[current_node, unvisited_nodes[i]]
        
        # Fill in DP
        for mask in range(1, full_mask + 1):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                if dp[mask, i] == float('inf'):
                    continue
                # Try to extend to node j not in mask
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = dp[mask, i] + distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
                    if new_cost < dp[new_mask, j]:
                        dp[new_mask, j] = new_cost
        
        # Find the best way to complete the tour: from any node i in full_mask, go to destination
        best_cost = float('inf')
        best_last = 0
        for i in range(n):
            total = dp[full_mask, i] + distance_matrix[unvisited_nodes[i], destination_node]
            if total < best_cost:
                best_cost = total
                best_last = i
        
        # Now we need to find the first node in the optimal path
        # We can backtrack, but for efficiency, let's just find which node to visit first
        # that leads to the optimal solution
        best_first = 0
        best_first_cost = float('inf')
        for i in range(n):
            # If we start with node i, the cost is dp[1<<i, i] + (cost to complete from i visiting all others)
            # This is equivalent to: dp[full_mask][i] + d(i, dest) but we need the first step
            # Actually, let's just pick the node that minimizes: d(current, v) + min completion from v
            # The min completion from v (having visited v, needing to visit all others) is:
            # min over j != v of dp[full_mask ^ (1<<v), j] + d(j, dest) ... but that's not quite right
            
            # Simpler: just evaluate each candidate first node v
            # Cost if first node is v: d(current, v) + min over paths visiting all remaining nodes
            # The min over paths from v visiting all other nodes then going to dest
            # This is: min over j of (dp[full_mask, j] where the path starts at v)
            
            # Let's just do it directly: for each candidate first node v,
            # compute the min cost to visit all other nodes starting from v and ending at any node,
            # then add d(last, dest)
            
            # This is essentially the same DP but starting from v
            # For efficiency with n<=8, let's just try all candidates
            for v_idx in range(n):
                v = unvisited_nodes[v_idx]
                # Cost to reach v from current
                cost_to_v = distance_matrix[current_node, v]
                
                # Now we need to visit all other nodes (n-1 nodes) starting from v
                # This is a TSP path problem with n-1 nodes
                # For n<=8, this is at most 7 nodes, so 7! = 5040 permutations, which is fine
                # But let's use DP again for efficiency
                
                if n == 2:
                    total = cost_to_v + distance_matrix[v, unvisited_nodes[1-v_idx]] + distance_matrix[unvisited_nodes[1-v_idx], destination_node]
                elif n == 3:
                    others = [unvisited_nodes[k] for k in range(n) if k != v_idx]
                    o1, o2 = others
                    total = cost_to_v + min(
                        distance_matrix[v, o1] + distance_matrix[o1, o2] + distance_matrix[o2, destination_node],
                        distance_matrix[v, o2] + distance_matrix[o2, o1] + distance_matrix[o1, destination_node]
                    )
                else:
                    # Use DP for the subproblem
                    # Nodes to visit: all except v_idx
                    sub_nodes = [unvisited_nodes[k] for k in range(n) if k != v_idx]
                    m = len(sub_nodes)
                    sub_full = (1 << m) - 1
                    sub_dp = np.full((1 << m, m), float('inf'))
                    
                    for i in range(m):
                        sub_dp[1 << i, i] = distance_matrix[v, sub_nodes[i]]
                    
                    for mask in range(1, sub_full + 1):
                        for i in range(m):
                            if not (mask & (1 << i)):
                                continue
                            if sub_dp[mask, i] == float('inf'):
                                continue
                            for j in range(m):
                                if mask & (1 << j):
                                    continue
                                new_mask = mask | (1 << j)
                                new_cost = sub_dp[mask, i] + distance_matrix[sub_nodes[i], sub_nodes[j]]
                                if new_cost < sub_dp[new_mask, j]:
                                    sub_dp[new_mask, j] = new_cost
                    
                    min_sub_cost = min(
                        sub_dp[sub_full, i] + distance_matrix[sub_nodes[i], destination_node]
                        for i in range(m)
                    )
                    total = cost_to_v + min_sub_cost
                
                if total < best_first_cost:
                    best_first_cost = total
                    best_first = v_idx
        
        return int(unvisited_nodes[best_first])

    # For n > 8, use a smarter heuristic
    # Strategy: consider each candidate, estimate the total tour cost if we pick that candidate
    # Use a combination of:
    # 1. Immediate cost: d(current, v)
    # 2. Connection cost: v needs one more edge. Estimate as min(d(v, nearest unvisited), d(v, dest))
    # 3. Future cost: approximate the cost of connecting all remaining nodes
    
    # Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Precompute min distance from each unvisited node to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For each candidate, compute a score
    # Score = d(current, v) + alpha * min_neighbor_cost(v) + beta * (avg remaining connection cost)
    
    # A good heuristic: the "2-approximation" style
    # When we pick v, v will have degree 2 in the final tour (one edge from current, one to somewhere)
    # The "somewhere" is either the next node in the tour or the destination (if v is last)
    
    # Let's use: score(v) = d(current, v) + 0.5 * min(d(v, u) for u in unvisited\{v}, d(v, dest))
    # This is the same as before but let's also consider a look-ahead
    
    best_node = -1
    best_score = float('inf')
    
    # For each candidate v, compute:
    # 1. Immediate: d(current, v)
    # 2. The best "exit" from v: min over u in unvisited\{v} of d(v,u), and d(v,dest)
    # 3. A rough estimate of the remaining tour cost
    
    # Precompute for each unvisited node, the min distance to any other unvisited node
    min_to_other = np.full(n, float('inf'))
    for i in range(n):
        v = unvisited_nodes[i]
        dists_v = distance_matrix[v, unvisited_nodes].copy()
        dists_v[i] = float('inf')
        min_to_other[i] = np.min(dists_v) if n > 1 else float('inf')
    
    for i in range(n):
        v = unvisited_nodes[i]
        immediate = dists_from_current[i]
        
        # Best exit from v
        exit_cost = min(min_to_other[i], dists_to_dest[i])
        
        # Total score
        score = immediate + 0.5 * exit_cost
        
        if score < best_score:
            best_score = score
            best_node = v
    
    # Additionally, do a 2-step look-ahead for the top few candidates
    top_k = min(5, n)
    # Get top k candidates by score
    scores = np.array([dists_from_current[i] + 0.5 * min(min_to_other[i], dists_to_dest[i]) for i in range(n)])
    top_indices = np.argsort(scores)[:top_k]
    
    best_la_score = float('inf')
    best_la_node = best_node
    
    for idx in top_indices:
        v = unvisited_nodes[idx]
        # Look ahead: after visiting v, what's the best next node?
        remaining = np.delete(unvisited_nodes, idx)
        if len(remaining) == 0:
            la_cost = dists_from_current[idx] + distance_matrix[v, destination_node]
        else:
            # Best next node after v
            dists_v = distance_matrix[v, remaining]
            next_best = remaining[np.argmin(dists_v)]
            next_cost = dists_v[np.argmin(dists_v)]
            
            # After visiting next_best, estimate the rest
            # Use a rough estimate: average of min distances
            remaining_after = np.delete(remaining, np.argmin(dists_v))
            if len(remaining_after) == 0:
                la_cost = dists_from_current[idx] + next_cost + distance_matrix[next_best, destination_node]
            else:
                # Rough estimate for the rest
                # Sum of min distances from each remaining node to its nearest neighbor
                est_rest = 0
                for r in remaining_after:
                    dr = distance_matrix[r, remaining_after]
                    dr[np.where(remaining_after == r)[0][0]] = float('inf')
                    est_rest += 0.5 * np.min(dr)
                # Plus connection to destination
                est_rest += 0.5 * distance_matrix[next_best, destination_node]
                la_cost = dists_from_current[idx] + next_cost + est_rest
        
        if la_cost < best_la_score:
            best_la_score = la_cost
            best_la_node = v
    
    return int(best_la_node)
