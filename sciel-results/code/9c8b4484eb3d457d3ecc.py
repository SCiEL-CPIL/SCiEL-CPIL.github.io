import random
import sys

def construct():
    # Vertices 0..39
    n = 40
    # Initialize with a known C6-free structure: 8 K5s
    # This gives 80 edges.
    edges = set()
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.add((5 * k + a, 5 * k + b))
    
    # Try to add edges using a greedy heuristic with C6 check
    # To speed up, we precompute adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Helper to check if adding (u,v) creates a C6
    # A C6 containing (u,v) would be u-v-w-x-y-z-u
    # Or u-v-w-x-y-z-u is not necessarily the form.
    # Any C6 using edge (u,v) must have path of length 4 from v to u not using v?
    # Path v - a - b - c - d - u.
    # We need to check if there exists a path of length 4 from v to u in G \ {(u,v)}
    # which does not use u or v as internal nodes?
    # Actually, if we add (u,v), a C6 is formed if there is a path of length 4 from u to v in the existing graph.
    # Path: u - a - b - c - d - v.
    # a,b,c,d must be distinct and not u,v.
    
    def has_path_len_4(u, v):
        # BFS/DFS from u, depth 4, avoid v until end?
        # We want path u-a-b-c-d-v.
        # a != v, b != u,v, c != u,v, d != u,v.
        # Also a,b,c,d distinct.
        
        # Optimization: 
        # Neighbors of u: N(u)
        # Neighbors of v: N(v)
        # We need a in N(u), d in N(v).
        # Path a - b - c - d of length 3.
        # b in N(a) \ {u}, c in N(b) \ {a}, d in N(c) \ {b}.
        # And d in N(v).
        
        # Iterate a in N(u)
        for a in range(n):
            if adj[u][a]:
                # a cannot be v (simple graph, no self loop, u!=v)
                if a == v: continue
                # Iterate b in N(a)
                for b in range(n):
                    if adj[a][b] and b != u:
                        # b cannot be a
                        # Iterate c in N(b)
                        for c in range(n):
                            if adj[b][c] and c != a:
                                # c cannot be b
                                # Check if c is connected to any d in N(v)
                                # such that d != c
                                # We need d in N(c) AND d in N(v).
                                # And d != c.
                                # Also d must not be u?
                                # Path u-a-b-c-d-v.
                                # d must be connected to v.
                                # d must be connected to c.
                                # d != c.
                                # d != u? If d=u, path is u-a-b-c-u-v? No, d is before v.
                                # If d=u, then u-a-b-c-u is a cycle?
                                # We need simple path.
                                # So d != u, d != a, d != b.
                                
                                # Find intersection of N(c) and N(v)
                                # Filter out c, u, a, b
                                found = False
                                for d in range(n):
                                    if adj[c][d] and adj[v][d]:
                                        if d == c or d == u or d == a or d == b:
                                            continue
                                        # Check if d is valid
                                        # d must be distinct from a,b,c,u,v
                                        if d == v: continue # d connected to v, but d != v for simple path u...v
                                        # Actually d is a vertex. d != v.
                                        return True
                                # If loop finishes, no d found for this c
                # End b
            # End a
        return False

    # Greedy addition
    # Candidate edges: all non-edges
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u][v]]
    random.shuffle(candidates)
    
    improved = True
    while improved:
        improved = False
        for u, v in candidates:
            if adj[u][v]:
                continue
            if not has_path_len_4(u, v):
                # Add edge
                edges.add((u, v))
                adj[u][v] = True
                adj[v][u] = True
                improved = True
                # Re-shuffle or continue? 
                # Adding an edge might block others, but doesn't unblock.
                # So we can just continue.
        
    return [list(e) for e in edges]
