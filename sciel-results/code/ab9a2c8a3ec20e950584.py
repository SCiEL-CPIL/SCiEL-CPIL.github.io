import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Precompute triplets? No, generate on the fly or use the local check.
    
    def is_k33_free_local(adj, u, v):
        # Check triplets containing u
        for a in range(n):
            if a == u: continue
            for b in range(a + 1, n):
                if b == u: continue
                # Triplet (u, a, b)
                common = adj[u] & adj[a] & adj[b]
                if common != 0:
                    c1 = common & (common - 1)
                    if c1 != 0:
                        c2 = c1 & (c1 - 1)
                        if c2 != 0:
                            return False
        # Check triplets containing v
        # Note: Triplets containing both u and v are checked in both loops.
        # This is fine, just redundant.
        for a in range(n):
            if a == v: continue
            for b in range(a + 1, n):
                if b == v: continue
                # Triplet (v, a, b)
                common = adj[v] & adj[a] & adj[b]
                if common != 0:
                    c1 = common & (common - 1)
                    if c1 != 0:
                        c2 = c1 & (c1 - 1)
                        if c2 != 0:
                            return False
        return True

    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_k33_free_local(adj, u, v):
                current_edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return best_edges
