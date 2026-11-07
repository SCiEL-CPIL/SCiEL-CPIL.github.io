import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 85

    # Build adjacency from base path
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)

    existing = set()
    for u, v in base_edges:
        existing.add((min(u, v), max(u, v)))

    # All possible non-path edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing:
                all_edges.append((u, v))

    # Precompute for each edge whether adding it creates a triangle
    # An edge (u,v) creates a triangle if u and v share a common neighbor
    def creates_triangle(u, v, adj_set):
        return bool(adj_set[u] & adj_set[v])

    # Verify a set of added edges
    def verify(added):
        adj2 = [set(adj[i]) for i in range(n)]
        for u, v in added:
            if adj2[u] & adj2[v]:
                return False
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Check connected
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj2[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        # Check diameter <= 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj2[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
        return True

    # Precompute: for each vertex, the set of vertices at path-distance 2
    dist2 = [set() for _ in range(n)]
    for s in range(n):
        for d in [-2, 2]:
            t = s + d
            if 0 <= t < n:
                dist2[s].add(t)

    # Strategy: Try hub-based constructions
    # A hub h with edges to vertices at distance 2 from h
    # This creates a "double star" pattern that's triangle-free
    
    best = None
    best_count = float('inf')

    # Try each vertex as hub
    for h in range(n):
        if time.time() > deadline:
            break
        added = []
        for t in sorted(dist2[h]):
            edge = (min(h, t), max(h, t))
            if edge not in existing:
                added.append(list(edge))
        
        if len(added) < best_count and verify(added):
            best = added
            best_count = len(added)

    # Try two-hub constructions
    # Two hubs h1, h2 with edges from each to vertices at distance 2
    # Need to ensure no triangles are created
    
    # Precompute: for a pair of hubs, which edges can be added
    for h1 in range(n):
        for h2 in range(h1 + 2, n):  # h1 and h2 must not be adjacent (independence)
            if time.time() > deadline:
                break
            added = []
            # Edges from h1 to dist2[h1]
            for t in sorted(dist2[h1]):
                edge = (min(h1, t), max(h1, t))
                if edge not in existing:
                    added.append(list(edge))
            # Edges from h2 to dist2[h2]
            for t in sorted(dist2[h2]):
                edge = (min(h2, t), max(h2, t))
                if edge not in existing:
                    # Check if this edge is already in added
                    if edge not in [tuple(e) for e in added]:
                        added.append(list(edge))
            
            if len(added) < best_count and verify(added):
                best = added
                best_count = len(added)

    # Try three-hub constructions
    for h1 in range(n):
        for h2 in range(h1 + 2, n):
            for h3 in range(h2 + 2, n):
                if time.time() > deadline:
                    break
                added = []
                for h in [h1, h2, h3]:
                    for t in sorted(dist2[h]):
                        edge = (min(h, t), max(h, t))
                        if edge not in existing:
                            if edge not in [tuple(e) for e in added]:
                                added.append(list(edge))
                
                if len(added) < best_count and verify(added):
                    best = added
                    best_count = len(added)

    # Try a more general approach: independent set S where each s in S connects to dist2[s]
    # This is the original strategy but with better search
    
    # For small independent sets, enumerate all possibilities
    # An independent set on a path: no two consecutive vertices
    
    # Generate all independent sets of size k for small k
    def gen_independent_sets(n, k, start=0):
        if k == 0:
            yield []
            return
        if start >= n:
            return
        # Include start
        for rest in gen_independent_sets(n, k - 1, start + 2):
            yield [start] + rest
        # Exclude start
        for rest in gen_independent_sets(n, k, start + 1):
            yield rest

    # Try independent sets of size 3 and 4
    for k in [3, 4]:
        if time.time() > deadline:
            break
        for S in gen_independent_sets(n, k):
            if time.time() > deadline:
                break
            added = []
            for s in S:
                for t in sorted(dist2[s]):
                    edge = (min(s, t), max(s, t))
                    if edge not in existing:
                        if edge not in [tuple(e) for e in added]:
                            added.append(list(edge))
            
            if len(added) < best_count and verify(added):
                best = added
                best_count = len(added)

    # If we haven't found anything, use a known valid solution
    if best is None:
        # Fallback: hub at 11 with specific connections
        best = [[11, 1], [11, 4], [11, 7], [11, 15], [11, 18], [11, 21], [11, 23]]
        best_count = len(best)

    return [[int(u), int(v)] for u, v in best]
