"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_diameter(n, adj):
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        max_dist = max(max_dist, max(dist))
    return max_dist

def _get_diameter(n, base_edges, added):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [list(s) for s in adj]
    return _bfs_diameter(n, adj_list)

def construct():
    start_time = time.time()
    n = P['n']
    base_edges = P['edges']
    base_set = set((min(e), max(e)) for e in base_edges)
    
    # Generate candidates: non-base edges, sorted by length descending
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in base_set]
    candidates.sort(key=lambda e: -(e[1] - e[0]))
    
    # Greedy: add edges that most reduce diameter, maintaining matching
    added = []
    used = set()
    current_diam = 23
    
    while current_diam > 4:
        if time.time() - start_time > 55:
            break
        best_edge = None
        best_diam = current_diam
        for u, v in candidates:
            if u in used or v in used:
                continue
            # Try adding this edge
            test_added = added + [(u, v)]
            d = _get_diameter(n, base_edges, test_added)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
            if time.time() - start_time > 58:
                break
        if best_edge is None:
            break
        added.append(best_edge)
        used.add(best_edge[0])
        used.add(best_edge[1])
        current_diam = best_diam
    
    # Verify
    if current_diam <= 4:
        return [[u, v] for u, v in added]
    else:
        return [[u, v] for u, v in added]
# EVOLVE-BLOCK-END
