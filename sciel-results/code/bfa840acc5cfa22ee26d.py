import itertools
from collections import deque

def construct():
    # Start from 8 disjoint copies of K4 (n=32, alpha=8, no C5)
    # Try to add edges between the 8 cliques to increase density
    # without creating a C5 and while keeping alpha <= 8
    
    n = 32
    # 8 blocks of 4 vertices each: block b has vertices 4*b .. 4*b+3
    # Initial edges: all edges within each block
    adj = [[False] * n for _ in range(n)]
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                adj[u][v] = adj[v][u] = True
    
    # Helper: check if adding edge (u,v) creates a C5
    # A C5 is formed if there's a path of length 3 from u to v in the current graph
    # that doesn't use u or v internally, and the path has 4 distinct vertices
    # Actually, adding edge (u,v) creates a C5 iff there exists a path of length 3
    # from u to v (i.e., u-x-y-v where x!=v, y!=u, x!=y) in the current graph.
    # More precisely: there exist distinct x, y such that adj[u][x], adj[x][y], adj[y][v] are all True,
    # and x != v, y != u, x != y, and x, y not in {u, v}.
    
    def creates_c5(u, v):
        # Check if there's a path u - x - y - v with all vertices distinct
        for x in range(n):
            if x == u or x == v:
                continue
            if not adj[u][x]:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if not adj[x][y]:
                    continue
                if adj[y][v]:
                    return True
        return False
    
    # Helper: compute independence number (upper bound check via greedy + exact for small n)
    # For n=32, exact alpha computation is expensive. We'll use a greedy upper bound.
    # Actually, we need to ensure alpha <= 8. Let's compute alpha exactly using branch and bound.
    
    def compute_alpha():
        # Compute independence number of the graph with adjacency matrix adj
        # Use a simple recursive backtracking with pruning
        # For n=32 this could be slow, so we use a heuristic approach:
        # The independence number is at least the size of the largest set we can greedily build
        # and we need to verify it's <= 8
        
        # Since we start from disjoint K4s, the initial alpha is 8 (one vertex from each block)
        # Adding edges between blocks can only decrease or maintain alpha
        # So if we start with alpha=8 and only add edges, alpha can only stay the same or decrease
        # Thus we just need to ensure we don't accidentally increase it, which can't happen when adding edges
        
        return 8  # Upper bound is still valid since we only add edges
    
    # Now try to add edges between blocks greedily
    # Generate all possible inter-block edges
    candidate_edges = []
    for b1 in range(8):
        for b2 in range(b1 + 1, 8):
            for i in range(4):
                for j in range(4):
                    u = 4 * b1 + i
                    v = 4 * b2 + j
                    candidate_edges.append((u, v))
    
    # Sort candidates: prioritize edges that connect vertices in different blocks
    # and try to add as many as possible without creating C5
    # Use a greedy approach with random restarts or deterministic ordering
    
    # Try each candidate edge in order, add if it doesn't create a C5
    for u, v in candidate_edges:
        if not adj[u][v] and not creates_c5(u, v):
            adj[u][v] = adj[v][u] = True
    
    # Build edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
