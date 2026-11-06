import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited <= 15:
        return _solve_held_karp(current_node, destination_node, unvisited_nodes, distance_matrix)
    else:
        return _solve_two_step_greedy(current_node, destination_node, unvisited_nodes, distance_matrix)

def _solve_held_karp(current, dest, unvisited, dist):
    k = len(unvisited)
    if k == 0:
        return dest
        
    # Map unvisited to 0..k-1
    # dp[mask][i] = min cost to start at current, visit set mask, end at unvisited[i]
    # mask is integer 0 to 2^k - 1
    # i is 0 to k-1
    
    INF = float('inf')
    # Using list of lists for speed
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]
    
    # Base cases: start at current, go to unvisited[i]
    for i in range(k):
        dp[1 << i][i] = dist[current, unvisited[i]]
        parent[1 << i][i] = -1
        
    # Fill DP table
    for mask in range(1 << k):
        for i in range(k):
            if dp[mask][i] == INF:
                continue
            if not (mask & (1 << i)):
                continue
            
            # Try adding j
            remaining = (~mask) & ((1 << k) - 1)
            # Iterate over bits in remaining
            r = remaining
            while r:
                lsb = r & (-r)
                j = lsb.bit_length() - 1
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + dist[unvisited[i], unvisited[j]]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
                    parent[new_mask][j] = i
                r -= lsb
                
    full_mask = (1 << k) - 1
    best_cost = INF
    best_last = -1
    for i in range(k):
        cost = dp[full_mask][i] + dist[unvisited[i], dest]
        if cost < best_cost:
            best_cost = cost
            best_last = i
            
    # Backtrack to find the first node
    if k == 1:
        return int(unvisited[0])
        
    # Reconstruct path
    path = []
    cur_mask = full_mask
    cur_i = best_last
    while cur_i != -1:
        path.append(cur_i)
        prev_i = parent[cur_mask][cur_i]
        if prev_i == -1:
            break
        cur_mask = cur_mask ^ (1 << cur_i)
        cur_i = prev_i
        
    # path is reversed: [last, ..., first]
    # The first node in the sequence is path[-1]
    first_idx = path[-1]
    return int(unvisited[first_idx])

def _solve_two_step_greedy(current, dest, unvisited, dist):
    # Fast heuristic: 2-step lookahead
    # Minimize dist(current, u) + min(dist(u, v) for v in unvisited \ {u})
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    # Vectorized computation for speed
    # dist_current: (n,)
    dist_current = dist[current, unvisited]
    
    # sub_dist: (n, n)
    sub_dist = dist[np.ix_(unvisited, unvisited)]
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(sub_dist, float('inf'))
    
    # min_next: (n,) - min distance from u to any other unvisited node
    min_next = np.min(sub_dist, axis=1)
    
    # Score: immediate cost + future cost estimate
    scores = dist_current + min_next
    
    # Find the node with the minimum score
    idx = np.argmin(scores)
    
    return int(unvisited[idx])
