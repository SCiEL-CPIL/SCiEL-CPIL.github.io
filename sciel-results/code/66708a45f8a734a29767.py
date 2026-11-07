"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, src, n):
    dist = [-1] * n
    dist[src] = 0
    queue = [src]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def _diameter(adj, n):
    max_d = 0
    for i in range(n):
        dist = _bfs_distances(adj, i, n)
        for d in dist:
            if d > max_d:
                max_d = d
    return max_d

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add(tuple(e))
    
    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    
    added = []
    
    def is_bipartite(u, v):
        return (u - v) % 2 != 0
    
    for _ in range(50):  # safety bound
        diam = _diameter(adj, n)
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = diam
        
        # Generate candidates: bipartite edges not in graph
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set and is_bipartite(u, v):
                    candidates.append((u, v))
        
        for u, v in candidates:
            adj[u].append(v)
            adj[v].append(u)
            
            d = _diameter(adj, n)
            
            adj[u].remove(v)
            adj[v].remove(u)
            
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        added.append([u, v])
    
    result = [[int(u), int(v)] for u, v in added]
    return result

# EVOLVE-BLOCK-END
