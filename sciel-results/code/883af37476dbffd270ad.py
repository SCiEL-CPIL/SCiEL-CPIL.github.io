def has_short_cycle_bfs(adj, g):
    n = len(adj)
    # We need to detect cycles of length < g
    # BFS from each node
    # For each node, keep track of (parent, depth)
    # If we reach a node at depth d with parent p, and we already reached it at depth d' with parent p'
    # If d == d' and p != p', cycle length 2d.
    # If d + d' < g, cycle exists.
    
    for start in range(n):
        # BFS
        # dist[v] = min depth
        # parent[v] = parent node
        # We need to detect if there are TWO distinct paths of length < g/2?
        # Actually, any cycle of length L < g will be found.
        
        visited = {start: (None, 0)}
        queue = [(start, None, 0)]
        
        while queue:
            u, parent_u, depth_u = queue.pop(0)
            if depth_u >= g - 1:
                continue
            for v in adj[u]:
                if v == parent_u:
                    continue
                if v not in visited:
                    visited[v] = (u, depth_u + 1)
                    queue.append((v, u, depth_u + 1))
                else:
                    # We found a connection to a visited node
                    # This indicates a cycle
                    # The cycle length is depth_u + 1 + depth_v
                    # But we need to be careful about backtracking
                    # Since we skip parent, this is a non-backtracking path
                    parent_v, depth_v = visited[v]
                    if parent_v == u:
                        # This is the back edge to parent, ignore? 
                        # No, we skipped v==parent_u. 
                        # If v is parent_u, we skipped it.
                        # So v is not parent_u.
                        # If u is parent_v, then we are going back?
                        # In BFS, if v is visited, and u is not parent_v, then we have a cycle.
                        # Cycle length: depth_u + depth_v + 1
                        cycle_len = depth_u + depth_v + 1
                        if cycle_len < g:
                            return True
                    else:
                        # u is not parent_v
                        # But we must ensure the paths are distinct
                        # In a tree, this doesn't happen.
                        # In a graph with cycles, it does.
                        cycle_len = depth_u + depth_v + 1
                        if cycle_len < g:
                            return True
    return False
