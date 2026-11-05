import time
import random
import sys

def construct():
    n = 40
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Start with 8 disjoint K5s
    adj = [[False] * n for _ in range(n)]
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
    
    def is_c6_free(adj):
        """Check if graph is C6-free. Used for validation."""
        # For each pair of non-adjacent vertices, check no path of length 5
        # This is expensive, so we only use it for final validation
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue
                # Check if there's a path of length 5 from u to v
                # Path: u - a - b - c - d - v
                for a in range(n):
                    if a == u or a == v or not adj[u][a]:
                        continue
                    for b in range(n):
                        if b == u or b == v or b == a or not adj[a][b]:
                            continue
                        for c in range(n):
                            if c == u or c == v or c == a or c == b or not adj[b][c]:
                                continue
                            for d in range(n):
                                if d == u or d == v or d == a or d == b or d == c or not adj[c][d]:
                                    continue
                                if adj[d][v]:
                                    return False
        return True
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) would create a C6."""
        # A C6 is created iff there's a path of length 4 from u to v
        # Path: u - x - y - z - v (4 edges, so 5 vertices including u,v)
        # Wait, adding (u,v) creates a C6 if there's a path of length 5 from u to v
        # Path: u - a - b - c - d - v (5 edges)
        for a in range(n):
            if a == u or a == v or not adj[u][a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a or not adj[a][b]:
                    continue
                for c in range(n):
                    if c == u or c == v or c == a or c == b or not adj[b][c]:
                        continue
                    for d in range(n):
                        if d == u or d == v or d == a or d == b or d == c or not adj[c][d]:
                            continue
                        if adj[d][v]:
                            return True
        return False
    
    def try_add_edge(adj, u, v, edges):
        if u == v or adj[u][v]:
            return False
        if would_create_c6(adj, u, v):
            return False
        adj[u][v] = adj[v][u] = True
        edges.append([u, v])
        return True
    
    def random_non_edge(adj):
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not adj[u][v]:
                return u, v
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    
    # Phase 1: Greedy add edges
    while time.time() < deadline - 5.0:
        progress = False
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    if try_add_edge(adj, u, v, edges):
                        progress = True
        if not progress:
            break
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    
    # Phase 2: Local search with random restarts of edge addition
    while time.time() < deadline:
        # Try random non-edges
        for _ in range(500):
            if time.time() >= deadline:
                break
            u, v = random_non_edge(adj)
            if try_add_edge(adj, u, v, edges):
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_adj = [row[:] for row in adj]
    
    # Return the best edges found
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                result.append([u, v])
    
    return result
