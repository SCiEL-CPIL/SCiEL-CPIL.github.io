import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base = P['edges']
    base_set = {tuple(sorted(e)) for e in base}
    deadline = time.time() + 80

    # Key insight: for a path, adding edge (u,v) creates a triangle iff |u-v| == 2.
    # So valid edges are those with |u-v| >= 3 (and not already in base).
    candidates = [(u, v) for u in range(n) for v in range(u+3, n) if (u, v) not in base_set]

    # Precompute path distances (the diameter lower bound)
    path_dist = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            path_dist[u][v] = abs(u - v)

    def bfs_dist(adj, src):
        """BFS from src, return list of distances."""
        dist = [-1]*n
        dist[src] = 0
        q = [src]
        idx = 0
        while idx < len(q):
            nd = q[idx]
            idx += 1
            d_next = dist[nd] + 1
            for nb in adj[nd]:
                if dist[nb] == -1:
                    dist[nb] = d_next
                    q.append(nb)
        return dist

    def check(added):
        """Check if base + added is triangle-free, connected, diameter <= 4."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in base:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check (should always pass given our candidate filter, but verify)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        
        # Diameter check via BFS from each vertex
        for src in range(n):
            dist = bfs_dist(adj, src)
            if -1 in dist:  # disconnected
                return False
            if max(dist) > 4:
                return False
        return True

    # Exhaustive search for small k
    for k in range(1, 12):
        if time.time() > deadline:
            break
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            if check(list(combo)):
                return [[int(u), int(v)] for u, v in combo]
    
    # Greedy fallback
    current_edges = list(base)
    cur_set = set(tuple(sorted(e)) for e in base)
    added = []
    
    def get_diameter(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        md = 0
        for src in range(n):
            dist = bfs_dist(adj, src)
            if -1 in dist:
                return 999
            md = max(md, max(dist))
        return md
    
    def is_tri_free(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u and (adj[u] & adj[v]):
                    return False
        return True
    
    cur_d = get_diameter(current_edges)
    while cur_d > 4 and time.time() < deadline:
        best_e = None
        best_d = cur_d + 1
        for (u, v) in candidates:
            if (u, v) in cur_set:
                continue
            ne = current_edges + [(u, v)]
            if not is_tri_free(ne):
                continue
            d = get_diameter(ne)
            if d < best_d:
                best_d = d
                best_e = (u, v)
        if best_e is None:
            break
        current_edges.append(best_e)
        cur_set.add(best_e)
        added.append(best_e)
        cur_d = best_d
    
    if cur_d <= 4:
        return [[int(u), int(v)] for u, v in added]
    
    return []
