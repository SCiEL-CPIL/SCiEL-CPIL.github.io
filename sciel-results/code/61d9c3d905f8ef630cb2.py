import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 100.0  # 100 seconds limit
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to check if adding edge (u, v) creates a 4-cycle
    # adj is a list of lists of neighbors
    def creates_c4(adj, u, v):
        # We need a path u-w-x-v with distinct u,v,w,x
        # adj[u] contains neighbors of u
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[w]:
                if x == u or x == v or x == w:
                    continue
                if v in adj[x]:
                    return True
        return False

    # Run multiple restarts
    # We can also try to be smarter about edge ordering in some restarts
    # e.g., random, or degree-based
    while time.time() < deadline:
        adj = [[] for _ in range(n)]
        edges = []
        
        # Shuffle edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if creates_c4(adj, u, v):
                continue
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we find a very good solution, we can continue to see if we can beat it,
            # but if we hit a known upper bound or a very high number, we might stop.
            # For n=20, z(20,2,2) is likely around 78-80.
            if best_count >= 80:
                break

    # Fallback if no edges found (unlikely)
    if not best_edges:
        return [[0, v, 0] for v in range(1, n)]

    # Assign colors
    # Since the graph is C4-free, the rainbow constraint is vacuously satisfied.
    # We just need to use colors 0..19.
    result = []
    for i, (u, v) in enumerate(best_edges):
        c = i % 20
        result.append([u, v, c])
        
    return result
