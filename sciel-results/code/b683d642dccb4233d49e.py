import random
import time

def construct():
    N = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Helper to check if adding edge (u, v) to graph (adj) creates K3,3
    def creates_k33(adj, u, v):
        # adj is a list of sets or lists of neighbors
        # Case 1: u and v in same part of K3,3
        # Need w such that |N(u) ∩ N(v) ∩ N(w)| >= 3
        common_uv = adj[u] & adj[v]
        if len(common_uv) >= 3:
            for w in range(N):
                if w == u or w == v:
                    continue
                # Check if w has at least 3 neighbors in common_uv
                # Since common_uv is a set, we can iterate or use set intersection
                # adj[w] & common_uv
                if len(adj[w] & common_uv) >= 3:
                    return True
        
        # Case 2: u and v in different parts
        # Need u2, u3 in N(v)\{u} and v2, v3 in N(u)\{v}
        # such that u2, u3 connected to v2, v3
        candidates_u = [x for x in adj[v] if x != u]
        candidates_v = [y for y in adj[u] if y != v]
        
        # Optimization: if len(candidates_u) < 2 or len(candidates_v) < 2, return False
        if len(candidates_u) < 2 or len(candidates_v) < 2:
            return False
            
        # Check all pairs in candidates_u
        for i in range(len(candidates_u)):
            u2 = candidates_u[i]
            for j in range(i+1, len(candidates_u)):
                u3 = candidates_u[j]
                # Count how many in candidates_v are connected to both u2 and u3
                count = 0
                for v2 in candidates_v:
                    if v2 in adj[u2] and v2 in adj[u3]:
                        count += 1
                        if count >= 2:
                            return True
        return False

    # Run random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random initialization
        p = 0.4
        adj = [set() for _ in range(N)]
        edge_set = set()
        
        for u, v in all_edges:
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        # Greedy addition
        # Shuffle missing edges
        missing_edges = [e for e in all_edges if e not in edge_set]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if not creates_k33(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        # Try a second pass? Sometimes order matters.
        # Let's try to add any remaining edges that might have become valid?
        # No, adding edges only increases constraints, so if it was invalid before, it's still invalid.
        # Wait, is that true?
        # If adding (u,v) created a K3,3, it was due to existing neighbors. Adding more edges doesn't remove those neighbors.
        # So once invalid, always invalid.
        # So one pass over shuffled edges is maximal for that shuffle.
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = list(edge_set)
            
        # If we reached a high number, we might stop early?
        # ex(20; K3,3) is likely around 100-150?
        # K2,18 is 36.
        # A random graph G(20, 0.5) has ~95 edges. It likely contains K3,3.
        # So we need to prune.
        
    # Convert best_edges to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
