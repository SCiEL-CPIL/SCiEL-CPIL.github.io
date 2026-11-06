import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_adjacency(edges, n):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def bfs_distances(adj, source, n):
        dist = {source: 0}
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        return dist, parent

    def find_path(parent, start, end):
        path = []
        cur = end
        while cur != start:
            path.append((cur, parent[cur]))
            cur = parent[cur]
        path.append((start, start)) # dummy to close loop? No, just edges
        # Reconstruct edges
        edges_in_path = []
        cur = end
        while cur != start:
            p = parent[cur]
            e = (min(cur, p), max(cur, p))
            edges_in_path.append(e)
            cur = p
        return edges_in_path

    def min_t_join_heuristic(edges, n):
        adj = get_adjacency(edges, n)
        
        # Find odd degree vertices
        odd_vertices = []
        for u in range(n):
            if len(adj[u]) % 2 != 0:
                odd_vertices.append(u)
        
        if not odd_vertices:
            return set()
        
        k = len(odd_vertices)
        
        # Compute all-pairs shortest paths between odd vertices
        # dist[i][j] is distance between odd_vertices[i] and odd_vertices[j]
        # paths[i][j] is the list of edges on one shortest path
        dist = [[0]*k for _ in range(k)]
        paths = [[[] for _ in range(k)] for _ in range(k)]
        
        for i, u in enumerate(odd_vertices):
            d, p = bfs_distances(adj, u, n)
            for j, v in enumerate(odd_vertices):
                if i == j:
                    dist[i][j] = 0
                elif v in d:
                    dist[i][j] = d[v]
                    paths[i][j] = find_path(p, u, v)
                else:
                    dist[i][j] = float('inf')
                    paths[i][j] = []

        # Greedy Matching + 2-Opt
        # State: list of pairs (i, j) where i < j
        # Initialize greedily
        used = [False] * k
        matches = []
        # Sort all pairs by distance
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                if dist[i][j] < float('inf'):
                    pairs.append((dist[i][j], i, j))
        pairs.sort()
        
        for d, i, j in pairs:
            if not used[i] and not used[j]:
                matches.append((i, j))
                used[i] = True
                used[j] = True
                if len(matches) == k // 2:
                    break
        
        if len(matches) != k // 2:
            # Fallback if greedy failed (shouldn't happen if connected)
            return set(edges) # Or just return empty and let rest be singles?
            # Actually, if graph is disconnected, T-Join might not exist in one component?
            # But instance_graph is connected (path + random edges).
            # So it should be fine.

        # 2-Opt Improvement
        def get_matching_weight(matches):
            w = 0
            for i, j in matches:
                w += dist[i][j]
            return w

        improved = True
        while improved:
            improved = False
            current_weight = get_matching_weight(matches)
            # Try swapping partners
            # For each pair of matches (a,b) and (c,d), try (a,c) and (b,d) or (a,d) and (b,c)
            for x in range(len(matches)):
                for y in range(x+1, len(matches)):
                    a, b = matches[x]
                    c, d = matches[y]
                    # Swap 1: (a,c) and (b,d)
                    if dist[a][c] < float('inf') and dist[b][d] < float('inf'):
                        new_w = current_weight - dist[a][b] - dist[c][d] + dist[a][c] + dist[b][d]
                        if new_w < current_weight:
                            matches[x] = (min(a,c), max(a,c))
                            matches[y] = (min(b,d), max(b,d))
                            improved = True
                            break
                    # Swap 2: (a,d) and (b,c)
                    if dist[a][d] < float('inf') and dist[b][c] < float('inf'):
                        new_w = current_weight - dist[a][b] - dist[c][d] + dist[a][d] + dist[b][c]
                        if new_w < current_weight:
                            matches[x] = (min(a,d), max(a,d))
                            matches[y] = (min(b,c), max(b,c))
                            improved = True
                            break
                if improved:
                    break
        
        # Reconstruct F (Symmetric Difference of Paths)
        F = set()
        for i, j in matches:
            for e in paths[i][j]:
                if e in F:
                    F.remove(e)
                else:
                    F.add(e)
        
        return F

    def decompose_eulerian(edges, n):
        adj = get_adjacency(edges, n)
        remaining = set(edges)
        cycles = []
        
        while remaining:
            # Find a start vertex with degree > 0
            start = None
            for u in range(n):
                if len(adj[u]) > 0:
                    start = u
                    break
            if start is None:
                break
                
            # Walk to find a cycle
            path = [start]
            current = start
            while True:
                # Find an unused edge from current
                next_v = None
                for v in adj[current]:
                    e = (min(current, v), max(current, v))
                    if e in remaining:
                        next_v = v
                        break
                if next_v is None:
                    break # Should not happen in Eulerian
                
                # Traverse edge
                e = (min(current, next_v), max(current, next_v))
                remaining.discard(e)
                adj[current].remove(next_v)
                adj[next_v].remove(current)
                
                current = next_v
                path.append(current)
                
                if current == start:
                    break
            
            if len(path) >= 3:
                cycles.append(path)
            else:
                # Should not happen if input is Eulerian and connected components are handled
                # But if we have a self-loop or 2-cycle?
                # Graph is simple, so no self-loops. No 2-cycles.
                # So cycles must be length >= 3.
                pass
                
        return cycles

    def construct_instance(edges, n):
        F = min_t_join_heuristic(edges, n)
        G_prime = set(edges) - F
        
        cycles = decompose_eulerian(list(G_prime), n)
        
        pieces = []
        pieces.extend(cycles)
        for e in F:
            pieces.append(list(e))
            
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        edges = instance(*spec)
        n = spec[0]
        
        # Run once, it's fast and deterministic
        pieces = construct_instance(edges, n)
        results.append(pieces)
        
    return results
