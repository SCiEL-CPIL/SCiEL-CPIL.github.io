import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 40
    
    def is_c6_free(adj):
        """Check if graph with adjacency matrix adj is C6-free."""
        # A 6-cycle: u0-u1-u2-u3-u4-u5-u0
        # For each u0, u1, u2, check if there's a path u2-u3-u4-u5-u0
        # with u3 != u0, u1, u2 and u4, u5 distinct and not u0,u1,u2
        # Use: for each (u0,u1,u2) with u1 in N(u0), u2 in N(u1), u2 != u0,
        #   check if N(u2) ∩ N(N(N(u0))) has appropriate structure
        # Simpler: count 6-cycles via matrix multiplication? Too slow.
        # Brute force with pruning:
        for u0 in range(n):
            for u1 in range(n):
                if u1 == u0 or not adj[u0, u1]:
                    continue
                for u2 in range(n):
                    if u2 in (u0, u1) or not adj[u1, u2]:
                        continue
                    for u3 in range(n):
                        if u3 in (u0, u1, u2) or not adj[u2, u3]:
                            continue
                        for u4 in range(n):
                            if u4 in (u0, u1, u2, u3) or not adj[u3, u4]:
                                continue
                            # Check if u5 exists: u5 in N(u4) ∩ N(u0), u5 not in {u0,u1,u2,u3,u4}
                            for u5 in range(n):
                                if u5 in (u0, u1, u2, u3, u4) or not adj[u4, u5] or not adj[u5, u0]:
                                    continue
                                return False
        return True
    
    # This brute force is O(n^6) = 40^6 = 4e9, way too slow.
    # Need a better approach.
    
    # Better: Use the fact that C6 exists iff there are u0, u2, u4 (even positions) 
    # such that N(u0) ∩ N(u2) != empty, N(u2) ∩ N(u4) != empty, N(u4) ∩ N(u0) != empty,
    # and the intersections are "compatible". 
    # Specifically: for u0, u2, u4 all distinct, let A = N(u0)∩N(u2), B = N(u2)∩N(u4), C = N(u4)∩N(u0).
    # If A, B, C are all non-empty and there exist a in A, b in B, c in C with a!=b!=c!=a and a!=c etc., 
    # we need to check more carefully.
    
    # Actually, a 6-cycle u0-u1-u2-u3-u4-u5-u0 requires:
    # u1 in N(u0)∩N(u2), u3 in N(u2)∩N(u4), u5 in N(u4)∩N(u0)
    # with u1, u3, u5 all distinct and not equal to u0,u2,u4.
    # So: for each triple (u0,u2,u4) of distinct vertices, check if 
    #   |N(u0)∩N(u2)| >= 1, |N(u2)∩N(u4)| >= 1, |N(u4)∩N(u0)| >= 1
    #   and the three intersection sets have a "triangle" of distinct elements.
    
    # Even simpler sufficient check: if for any distinct u0,u2,u4, 
    #   (N(u0)∩N(u2)) ∩ (N(u4)∩N(u0)) etc. ... this is getting messy.
    
    # Let me use a different strategy: since n=40 is small, use a precomputed 
    # known 105-edge graph.
    
    # Known construction: See "Extremal C6-free graphs" literature.
    # One such graph is the "Hoffman-Singleton" subgraph or a specific SRG.
    
    # Let me try: start with K40, remove edges to break all C6s greedily.
    # This is expensive but might work with a good heuristic.
    
    # Alternative: use the complement of a C6-free graph? No.
    
    # Let me try a random greedy construction:
    random.seed(42)
    
    # Start with no edges
    adj = np.zeros((n, n), dtype=bool)
    edges = []
    
    # Try adding edges in a random order, keeping C6-freedom
    # But checking C6-freedom after each edge is too slow.
    
    # Instead, use a structured seed: 8 copies of K5 (80 edges), then try to add edges.
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u, v = 5*k+a, 5*k+b
                adj[u, v] = adj[v, u] = True
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try to add cross edges between the K5 components
    # Check if adding edge (u,v) creates a C6
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) to graph adj creates a C6."""
        # A new C6 must use the new edge (u,v).
        # So we need a path of length 5 from u to v in the current graph.
        # Path: u - a - b - c - d - v, with a,b,c,d distinct and not u,v.
        # a in N(u), b in N(a)∩N(?), etc.
        # Specifically: a in N(u), b in N(a)\{u}, c in N(b)\{a,u}, d in N(c)\{b,a,u}, v in N(d)\{c,b,a,u}
        # and d in N(v).
        # So: for a in N(u), for b in N(a)\\{u}, for c in N(b)\\{a,u}, for d in N(c)\\{b,a,u}, check if d in N(v) and d != v
        for a in range(n):
            if a == u or not adj[u, a]:
                continue
            for b in range(n):
                if b in (u, a) or not adj[a, b]:
                    continue
                for c in range(n):
                    if c in (u, a, b) or not adj[b, c]:
                        continue
                    for d in range(n):
                        if d in (u, a, b, c) or not adj[c, d]:
                            continue
                        if d == v or not adj[d, v]:
                            continue
                        return True
        return False
    
    # Try to add all possible cross edges
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                continue
            if time.time() > deadline:
                return best_edges
            if not would_create_c6(adj, i, j):
                adj[i, j] = adj[j, i] = True
                edges.append([i, j])
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
    
    return best_edges
