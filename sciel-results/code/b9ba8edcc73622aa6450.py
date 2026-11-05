import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding edge (u, v) to graph G is valid
    # G is a list of sets of neighbors
    def can_add(G, u, v):
        # 1. Check common neighbors of u and v
        # If |N(u) ∩ N(v)| >= 3, adding (u,v) creates K2,3 with {u,v} as the pair
        common_uv = G[u] & G[v]
        if len(common_uv) >= 3:
            return False
        
        # 2. Check if adding (u,v) creates a K2,3 with {u, w} as the pair for some w
        # This happens if v is a neighbor of w, and u and w already have 2 common neighbors.
        # Then v becomes the 3rd common neighbor.
        # So for all w in N(v) \ {u}, we need |N(u) ∩ N(w)| <= 1
        for w in G[v]:
            if w == u:
                continue
            # w is a neighbor of v
            # Check intersection of N(u) and N(w)
            # Note: v is not in N(u) yet, so N(u) is current N(u)
            # w is in N(v), so v is in N(w)
            # We are checking if u and w have >= 2 common neighbors in current graph
            # If they do, adding (u,v) makes v a common neighbor too, total 3.
            common_uw = G[u] & G[w]
            if len(common_uw) >= 2:
                return False
                
        # 3. Check if adding (u,v) creates a K2,3 with {v, w} as the pair for some w
        # Symmetric to above
        for w in G[u]:
            if w == v:
                continue
            common_vw = G[v] & G[w]
            if len(common_vw) >= 2:
                return False
                
        return True

    # Greedy construction with random restarts
    for seed in range(1000):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        # Graph representation: list of sets
        G = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in edges_to_try:
            if can_add(G, u, v):
                G[u].add(v)
                G[v].add(u)
                current_edges.append((u, v))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    # Convert to list of lists [u, v] as required
    return [[u, v] for u, v in best_edges]
