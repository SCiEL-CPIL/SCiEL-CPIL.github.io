"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # Build adjacency for base path
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: only pairs (u,v) where u<v, u-v is odd (even-odd pairs), and not in base
    # Since base is a path, consecutive vertices have difference 1 (odd), so they're already in base
    # We need even-odd pairs that are NOT consecutive
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 != 0 and (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Helper: check if adding edge (u,v) to current_adj creates a triangle
    def creates_triangle(current_adj, u, v):
        # Triangle if there exists w adjacent to both u and v
        return bool(current_adj[u] & current_adj[v])
    
    # Helper: BFS diameter
    def diameter(current_adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nei in current_adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            max_d = max(max_d, max(dist))
        return max_d
    
    # Helper: validate a set of added edges
    def validate(added_edges):
        # Build full edge set
        all_edges = set(base_edges)
        for u, v in added_edges:
            if u > v:
                u, v = v, u
            if (u, v) in all_edges:
                return False  # duplicate
            all_edges.add((u, v))
        
        # Build adjacency
        v_adj = [set() for _ in range(n)]
        for u, v in all_edges:
            v_adj[u].add(v)
            v_adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            for v in v_adj[u]:
                if u < v:  # avoid double-check
                    if v_adj[u] & v_adj[v]:
                        return False
        
        # Check connected
        visited = set()
        q = deque([0])
        visited.add(0)
        while q:
            node = q.popleft()
            for nei in v_adj[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append(nei)
        if len(visited) != n:
            return False
        
        # Check diameter <= 4
        if diameter(v_adj) > 4:
            return False
        
        return True
    
    # Try to find minimum number of added edges
    # Start with k=0,1,2,... until we find a valid solution
    deadline = time.time() + 55  # 55 second safety margin
    
    # First, check if base path already satisfies constraints
    # Base path: connected? yes. Triangle-free? yes. Diameter? 23 > 4. No.
    
    for k in range(0, 10):  # try up to 9 added edges
        if time.time() > deadline:
            break
        
        if k == 0:
            if validate([]):
                return []
            continue
        
        if k == 1:
            # Try each single edge
            for (u, v) in candidates:
                if time.time() > deadline:
                    break
                if creates_triangle(adj, u, v):
                    continue
                test_adj = [set(s) for s in adj]
                test_adj[u].add(v)
                test_adj[v].add(u)
                if validate([(u, v)]):
                    return [[int(u), int(v)]]
            continue
        
        if k == 2:
            # Try pairs of edges
            # Optimize: pre-filter candidates that don't create triangles with base
            valid_single = []
            for (u, v) in candidates:
                if not creates_triangle(adj, u, v):
                    valid_single.append((u, v))
            
            for i in range(len(valid_single)):
                if time.time() > deadline:
                    break
                u1, v1 = valid_single[i]
                for j in range(i + 1, len(valid_single)):
                    u2, v2 = valid_single[j]
                    # Check if both edges can be added without creating triangles
                    # Check triangle for edge 1: adj[u1] & adj[v1] already checked
                    # Check triangle for edge 2: adj[u2] & adj[v2] already checked
                    # But adding both might create new triangles involving both edges
                    # Actually, a triangle needs 3 edges. If we add (u1,v1) and (u2,v2),
                    # a new triangle would need a third edge from base or the other added edge
                    # Let's just validate directly
                    if validate([(u1, v1), (u2, v2)]):
                        return [[int(u1), int(v1)], [int(u2), int(v2)]]
            continue
        
        # For k >= 3, use greedy search
        current_added = []
        current_adj = [set(s) for s in adj]
        current_edges_set = set(base_edges)
        
        for _ in range(k):
            if time.time() > deadline:
                break
            best_edge = None
            best_diam = float('inf')
            
            for (u, v) in candidates:
                if (u, v) in current_edges_set:
                    continue
                if creates_triangle(current_adj, u, v):
                    continue
                # Try adding this edge
                test_adj = [set(s) for s in current_adj]
                test_adj[u].add(v)
                test_adj[v].add(u)
                d = diameter(test_adj)
                if d < best_diam:
                    best_diam = d
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            current_adj[best_edge[0]].add(best_edge[1])
            current_adj[best_edge[1]].add(best_edge[0])
            current_edges_set.add(best_edge)
            current_added.append(best_edge)
        
        if len(current_added) == k:
            if validate(current_added):
                return [[int(u), int(v)] for u, v in current_added]
    
    # Fallback: return empty list (should not happen for valid instances)
    return []
# EVOLVE-BLOCK-END
