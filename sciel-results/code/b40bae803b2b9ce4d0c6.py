import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen if called correctly, but safe guard
        return -1
        
    if n_unvisited <= 15:
        return _solve_held_karp(current_node, destination_node, unvisited_nodes, distance_matrix)
    else:
        return _solve_lookahead_nearest(current_node, destination_node, unvisited_nodes, distance_matrix)

def _solve_held_karp(current, dest, unvisited, dist):
    k = len(unvisited)
    if k == 0:
        return dest # Or -1, depends on contract. If no unvisited, we go to dest? 
        # The function returns the NEXT node. If no unvisited, the next node is dest?
        # Usually, the loop stops when unvisited is empty.
        # Let's assume if unvisited is empty, we don't call this.
        
    # Map unvisited to 0..k-1
    # dp[mask][i] = min cost to start at current, visit set mask, end at unvisited[i]
    # mask is integer 0 to 2^k - 1
    # i is 0 to k-1
    
    INF = float('inf')
    # Initialize dp
    # Using list of lists for speed
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]
    
    # Base cases: start at current, go to unvisited[i]
    for i in range(k):
        dp[1 << i][i] = dist[current, unvisited[i]]
        parent[1 << i][i] = -1 # No previous unvisited node
        
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
    # The path is current -> ... -> unvisited[first] -> ... -> unvisited[best_last] -> dest
    # We need the first node after current.
    
    # If k == 1, first is 0
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

def _solve_lookahead_nearest(current, dest, unvisited, dist):
    # For each candidate u in unvisited:
    # cost = dist[current, u] + dist[u, nearest_neighbor(u, unvisited \ {u}, dest?)]
    # Actually, simple lookahead: dist[current, u] + dist[u, v] where v is the nearest to u among unvisited\{u}
    # If no other unvisited, use dist[u, dest]
    
    best_node = -1
    best_score = float('inf')
    
    for u in unvisited:
        d1 = dist[current, u]
        
        # Find nearest neighbor of u in unvisited excluding u
        # If only one node left, look at dest
        if len(unvisited) == 1:
            d2 = dist[u, dest]
        else:
            # Get distances from u to all unvisited
            # We can just loop or use argmin
            # unvisited is np array
            # dist[u, unvisited] gives array
            # We need to exclude u itself.
            # Since u is in unvisited, dist[u, u] is 0.
            # We want min dist[u, v] for v != u.
            # We can set dist[u, u] to inf temporarily or just mask it.
            
            # Efficient way:
            # Create a copy of distances? No, just find min excluding u.
            # np.argmin will pick u (0).
            # So we need to find the 2nd smallest? Or mask u.
            
            # Let's just iterate if k is small, or use a trick.
            # Since n_unvisited > 15, k is large.
            # dist[u, unvisited] is a vector.
            # We can set the element corresponding to u to inf.
            # But we don't know the index of u in unvisited easily without search.
            # unvisited is a np array.
            
            # Simpler:
            # d_vec = dist[u, unvisited]
            # d_vec[d_vec == 0] = inf # assumes no self loops with 0 dist other than u
            # But there might be other 0 dists? Unlikely in Euclidean.
            # Better:
            # Find index of u in unvisited
            # idx = np.where(unvisited == u)[0][0]
            # d_vec = dist[u, unvisited].copy()
            # d_vec[idx] = inf
            # d2 = np.min(d_vec)
            
            # This is O(N) per candidate, total O(N^2).
            # N=50, 2500 ops. Fine.
            
            idx_u = np.where(unvisited == u)[0][0]
            d_vec = dist[u, unvisited].copy()
            d_vec[idx_u] = float('inf')
            d2 = np.min(d_vec)
            
        score = d1 + d2
        if score < best_score:
            best_score = score
            best_node = u
            
    return int(best_node)
