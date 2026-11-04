def check_girth_voltage_fast(p, gens, g):
    n = 2 * p
    adj = [[] for _ in range(n)]
    for u in range(p):
        for g_val in gens:
            v_target = (u + g_val) % p
            adj[2*u].append(2*v_target + 1)
            adj[2*v_target + 1].append(2*u)
            
    max_depth = g // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # We need to ensure it's not a back-edge to parent
                    # In undirected graph BFS, if v is visited and v != parent[u], it's a cycle.
                    # But we don't track parent here.
                    # However, if dist[v] <= dist[u], and v is visited, it forms a cycle.
                    # Actually, any edge (u,v) where both are visited and v is not the one we came from?
                    # In BFS tree, non-tree edges connect nodes at same or adjacent levels.
                    # Cycle length = dist[u] + dist[v] + 1.
                    # If this is < g, return False.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
