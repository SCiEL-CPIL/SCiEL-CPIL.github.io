"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edge_list):
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj):
    for u in range(n):
        for v in list(adj[u]):
            if v > u:
                if adj[u] & adj[v]:
                    return False
    return True

def _bfs_diameter(n, adj):
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    if max_dist > 4:
                        return max_dist
                    q.append(v)
    return max_dist

def _valid_solution(n, added_edges, base_edges):
    all_edges = list(base_edges) + added_edges
    adj = _build_adj(n, all_edges)
    # Check connected
    visited = set([0])
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    if len(visited) != n:
        return False
    if not _is_triangle_free(n, adj):
        return False
    d = _bfs_diameter(n, adj)
    if d > 4:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    base_edges = []
    base_edge_set = set()
    for u, v in P['edges']:
        base_edges.append((u, v))
        base_edge_set.add((u, v))
        base_edge_set.add((v, u))
    
    base_adj = _build_adj(n, base_edges)
    
    # Find all valid candidate edges (even-odd, not in base, no common neighbor in base)
    candidate_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if u % 2 == v % 2:
                continue
            if (u, v) in base_edge_set:
                continue
            if base_adj[u] & base_adj[v]:
                continue
            candidate_edges.append((u, v))
    
    best_solution = None
    best_count = n + 1
    
    # Try 1 hub: each vertex as hub, connect to all opposite-parity candidates
    for hub in range(n):
        if time.time() > deadline:
            break
        hub_edges = []
        for c in candidate_edges:
            if hub in c:
                hub_edges.append(list(c))
        if len(hub_edges) >= best_count:
            continue
        if _valid_solution(n, hub_edges, base_edges):
            best_count = len(hub_edges)
            best_solution = hub_edges
    
    # Try 2 hubs
    for h1 in range(n):
        if time.time() > deadline:
            break
        for h2 in range(h1+1, n):
            if time.time() > deadline:
                break
            hub_edges = []
            for c in candidate_edges:
                if c[0] in (h1, h2) or c[1] in (h1, h2):
                    hub_edges.append(list(c))
            if len(hub_edges) >= best_count:
                continue
            if _valid_solution(n, hub_edges, base_edges):
                best_count = len(hub_edges)
                best_solution = hub_edges
    
    # Try 3 hubs
    for h1 in range(n):
        if time.time() > deadline:
            break
        for h2 in range(h1+1, n):
            if time.time() > deadline:
                break
            for h3 in range(h2+1, n):
                if time.time() > deadline:
                    break
                hub_edges = []
                for c in candidate_edges:
                    if c[0] in (h1, h2, h3) or c[1] in (h1, h2, h3):
                        hub_edges.append(list(c))
                if len(hub_edges) >= best_count:
                    continue
                if _valid_solution(n, hub_edges, base_edges):
                    best_count = len(hub_edges)
                    best_solution = hub_edges
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: single hub at 12
    added = []
    for v in range(n):
        if v % 2 != 0:
            if (12, v) not in base_edge_set and (v, 12) not in base_edge_set:
                added.append([12, v])
    return added
# EVOLVE-BLOCK-END
