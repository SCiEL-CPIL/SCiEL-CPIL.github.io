"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_triangle_free(adj, n):
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u]:
                if w > v and w in adj[v]:
                    return False
    return True

def _bfs_diameter(adj, n):
    max_diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d > max_diam:
                max_diam = d
    return max_diam

def _is_valid(added_edges, n, base_edges):
    all_edges = list(base_edges) + added_edges
    adj = _build_adj(all_edges, n)
    # Check connectivity
    if _bfs_diameter(adj, n) == -1:
        return False
    # Check triangle-free
    if not _check_triangle_free(adj, n):
        return False
    # Check diameter
    diam = _bfs_diameter(adj, n)
    if diam > 4:
        return False
    return True

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for e in base_edges:
        base_set.add((min(e), max(e)))
    
    # All possible edges not in base
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                all_possible.append((u, v))
    
    # Try small numbers of edges first
    from itertools import combinations
    
    deadline = time.time() + 80  # Leave 10s margin
    
    # Try 0 edges (should fail)
    # Try 1 edge
    # ... up to some limit
    
    best = None
    best_count = len(all_possible) + 1
    
    # Greedy approach: try to find a valid solution with few edges
    # Start with a heuristic: add edges that reduce diameter the most
    
    # Let's try a structured approach: 
    # For diameter 4 on 24 vertices, we need significant shortcuts
    # Try adding edges in a pattern
    
    # First, let's try a known good pattern
    # Add edges that create a "jump" structure
    candidate_sets = []
    
    # Pattern 1: Add edges (0,5), (5,10), (10,15), (15,20), (20,23)
    # But need to check triangle-free
    
    # Let me try a different approach: use a greedy algorithm
    # Start with base graph, greedily add edges that reduce max distance
    
    adj = _build_adj(base_edges, n)
    current_edges = list(base_edges)
    
    # Compute current distances
    def get_max_dist(adj, n):
        max_d = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    current_max = get_max_dist(adj, n)
    
    # Greedy: repeatedly add the edge that most reduces the max distance
    added = []
    used = set(base_set)
    
    while current_max > 4 and time.time() < deadline:
        best_edge = None
        best_new_max = current_max + 1
        
        for u, v in all_possible:
            if (u, v) in used:
                continue
            # Check if adding this edge creates a triangle
            # Triangle would be formed if there's a w such that (u,w) and (v,w) are both in graph
            # Actually, we need to check if u and v share a common neighbor
            common = adj[u] & adj[v]
            if common:
                continue  # Would create triangle
            
            # Add edge and check
            adj[u].add(v)
            adj[v].add(u)
            new_max = get_max_dist(adj, n)
            adj[u].discard(v)
            adj[v].discard(u)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        used.add((u, v))
        added.append([u, v])
        current_max = best_new_max
    
    # Check if valid
    if current_max <= 4 and _check_triangle_free(adj, n):
        return added
    
    # Fallback: try exhaustive search for small k
    for k in range(1, 10):
        if time.time() > deadline:
            break
        for combo in combinations(all_possible, k):
            if time.time() > deadline:
                break
            test_edges = base_edges + [list(c) for c in combo]
            adj_test = _build_adj(test_edges, n)
            if _check_triangle_free(adj_test, n):
                diam = _bfs_diameter(adj_test, n)
                if diam <= 4:
                    return [list(c) for c in combo]
    
    # If we get here, return what we have (may be invalid, but best effort)
    return added
# EVOLVE-BLOCK-END
