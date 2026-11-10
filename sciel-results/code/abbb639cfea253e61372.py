import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 16
    # Partition vertices into two sets of 8
    # A: 0-7, B: 8-15
    # Complete bipartite graph K_8,8
    # This graph is triangle-free.
    # Diameter is 2.
    # Every edge is critical because removing any edge (u,v) with u in A, v in B
    # increases the distance between u and v to 3 (via A-B-A or B-A-B paths don't exist directly, 
    # wait, u connects to B, v connects to A. Path u-b-a-v is length 3).
    # So diameter becomes 3.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
            
    # Verify types are strictly Python int and list
    for e in edges:
        assert isinstance(e, list), "Edge must be a list"
        assert len(e) == 2, "Edge must have 2 elements"
        assert isinstance(e[0], int) and not isinstance(e[0], bool), "Vertex must be int, not bool"
        assert isinstance(e[1], int) and not isinstance(e[1], bool), "Vertex must be int, not bool"
    
    # Quick verification that diameter is 2 and every edge is critical
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check diameter is exactly 2
    def bfs_diameter(adj, n):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            qi = 0
            while qi < len(queue):
                cur = queue[qi]
                qi += 1
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
                        if dist[nb] > max_dist:
                            max_dist = dist[nb]
            if -1 in dist:
                return -1  # disconnected
        return max_dist
    
    if time.time() < deadline:
        diam = bfs_diameter(adj, n)
        if diam != 2:
            # Fallback: return empty list if verification fails (shouldn't happen)
            return []
        
        # Verify each edge is critical
        for i, (u, v) in enumerate(edges):
            if time.time() > deadline:
                break
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            new_diam = bfs_diameter(adj, n)
            
            # Restore edge
            adj[u].add(v)
            adj[v].add(u)
            
            if new_diam <= 2:
                # Edge is not critical - this shouldn't happen for K_8,8
                # But if it does, we'd need to handle it. For K_8,8, this won't happen.
                pass
    
    return edges
