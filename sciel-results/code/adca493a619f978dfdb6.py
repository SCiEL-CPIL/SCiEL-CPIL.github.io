"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(adj):
    n = P['n']
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    queue.append(v)
    return max_dist

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    adj = _build_adj(P['edges'])
    
    # Bipartite: even and odd
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # Generate candidates sorted by path distance (longer first)
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (min(u,v), max(u,v)) not in base_edges:
                path_dist = abs(u - v)
                candidates.append((path_dist, u, v))
    candidates.sort(reverse=True)
    
    added = []
    current_adj = [set(s) for s in adj]
    used = set()
    
    while True:
        diam = _bfs_diameter(current_adj)
        if diam <= 4:
            break
        
        best = None
        for (pd, u, v) in candidates:
            if (u, v) in used or (v, u) in used:
                continue
            current_adj[u].add(v)
            current_adj[v].add(u)
            nd = _bfs_diameter(current_adj)
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            if best is None or nd < best[0]:
                best = (nd, u, v)
            if best[0] <= 4:
                break
        
        if best is None:
            break
        _, u, v = best
        current_adj[u].add(v)
        current_adj[v].add(u)
        added.append([u, v])
        used.add((u, v))
        used.add((v, u))
    
    return added
# EVOLVE-BLOCK-END

