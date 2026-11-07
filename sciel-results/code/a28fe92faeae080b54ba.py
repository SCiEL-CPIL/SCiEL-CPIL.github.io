"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, edges):
    adj = _build_adj(n, edges)
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
    return True

def _bfs_diameter(n, edges):
    adj = _build_adj(n, edges)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    queue.append(v)
    return max_dist

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set((min(u,v), max(u,v)) for u, v in base_edges)
    
    # Bipartite candidates: even-odd non-adjacent
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in base_edge_set:
                continue
            if (u % 2) == (v % 2):
                continue
            candidates.append((u, v))
    
    # Start with all candidates, greedily remove
    added = list(candidates)
    
    # Verify initial validity
    all_edges = list(base_edges) + added
    assert _is_triangle_free(n, all_edges)
    diam = _bfs_diameter(n, all_edges)
    assert diam <= 4
    
    # Greedy removal
    for edge in list(added):
        if time.time() > deadline:
            break
        new_added = [e for e in added if e != edge]
        new_edges = list(base_edges) + new_added
        if _is_triangle_free(n, new_edges) and _bfs_diameter(n, new_edges) <= 4:
            added = new_added
    
    return [[u, v] for u, v in added]

# EVOLVE-BLOCK-END

