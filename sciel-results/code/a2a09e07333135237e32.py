"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_all(n, adj):
    """Compute all-pairs shortest path distances."""
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[src][v] == -1:
                    dist[src][v] = dist[src][u] + 1
                    q.append(v)
    return dist

def _is_valid(n, base_edges, added_edges, max_diam=4):
    """Check triangle-free, connected, diameter <= max_diam."""
    all_edges = list(base_edges) + list(added_edges)
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = _build_adj(n, all_edges)
    
    # Connected
    visited = [False] * n
    visited[0] = True
    q = deque([0])
    cnt = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                cnt += 1
                q.append(v)
    if cnt != n:
        return False
    
    # Triangle-free
    for u, v in all_edges:
        nu = set(adj[u])
        nv = set(adj[v])
        if nu & nv:
            return False
    
    # Diameter
    dist = _bfs_all(n, adj)
    for i in range(n):
        for j in range(n):
            if dist[i][j] > max_diam:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 85.0
    n = P['n']
    base_edges = P['edges']
    base_set = set((min(u,v), max(u,v)) for u, v in base_edges)
    
    # Candidate edges: only even-odd pairs not in base, and no common neighbor in base path
    # In a path, vertices i and j (i<j) share a neighbor iff |i-j|==2 (they share i+1 or j-1)
    # So valid candidates: even-odd pairs with |i-j| >= 3 and not already in base
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i % 2) == (j % 2):
                continue
            key = (i, j)
            if key in base_set:
                continue
            # Check no common neighbor in the path
            # Neighbors of i in path: i-1, i+1 (if exist)
            # Neighbors of j in path: j-1, j+1 (if exist)
            common = set()
            if i > 0:
                common.add(i-1)
            if i < n-1:
                common.add(i+1)
            neigh_i = common
            common2 = set()
            if j > 0:
                common2.add(j-1)
            if j < n-1:
                common2.add(j+1)
            if neigh_i & common2:
                continue
            candidates.append((i, j))
    
    # Start with a known valid solution (hub-based)
    # Try to find a good initial solution
    best_solution = None
    best_count = float('inf')
    
    # Initial: use a structured pattern
    # For diameter 4 on 24 vertices, we need roughly 24/4 = 6 "hubs" or similar
    # Let's try adding edges to create a 4-regular-ish structure
    
    # Strategy: try all subsets of candidate edges of increasing size
    # But that's too large. Instead, use greedy + local search
    
    # Greedy: add edges that reduce the maximum distance the most
    def greedy_construct():
        added = []
        all_e = list(base_edges)
        for _ in range(20):
            if time.time() > deadline:
                break
            adj = _build_adj(n, all_e)
            dist = _bfs_all(n, adj)
            max_d = 0
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > max_d:
                        max_d = dist[i][j]
            if max_d <= 4:
                break
            
            best_edge = None
            best_max_d = max_d + 1
            for (u, v) in candidates:
                key = (min(u,v), max(u,v))
                if key in set((min(x[0],x[1]), max(x[0],x[1])) for x in added):
                    continue
                test_e = all_e + [[u, v]]
                # Quick check: triangle-free
                adj_u = set(adj[u])
                adj_v = set(adj[v])
                if adj_u & adj_v:
                    continue
                # Full check
                if not _is_valid(n, base_edges, added + [[u, v]], 4):
                    continue
                t_adj = _build_adj(n, test_e)
                t_dist = _bfs_all(n, t_adj)
                t_max = 0
                for i in range(n):
                    for j in range(n):
                        if t_dist[i][j] > t_max:
                            t_max = t_dist[i][j]
                if t_max < best_max_d:
                    best_max_d = t_max
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            added.append(list(best_edge))
            all_e.append(list(best_edge))
        
        return added
    
    added = greedy_construct()
    
    # Local search: try to remove edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(added)):
            test = added[:i] + added[i+1:]
            if _is_valid(n, base_edges, test, 4):
                added = test
                improved = True
                break
    
    # Try to swap: remove one, add another
    for _ in range(3):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(added)):
            if time.time() > deadline:
                break
            for (u, v) in candidates:
                key_new = (min(u,v), max(u,v))
                if key_new in set((min(x[0],x[1]), max(x[0],x[1])) for x in added):
                    continue
                test = added[:i] + added[i+1:] + [[u, v]]
                if len(test) < len(added):
                    continue
                if _is_valid(n, base_edges, test, 4) and len(test) < len(added):
                    added = test
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Final verification
    if not _is_valid(n, base_edges, added, 4):
        # Fallback
        added = greedy_construct()
    
    return [[int(u), int(v)] for u, v in added]
# EVOLVE-BLOCK-END
