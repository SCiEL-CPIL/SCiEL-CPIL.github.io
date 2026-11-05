import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initialize with Clebsch graph
    adj = [0] * n
    for u in range(n):
        for bit_idx in range(4):
            v = u ^ (1 << bit_idx)
            if u < v:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        v = u ^ 15
        if u < v:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    def edge_count(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def is_3_colorable(adj):
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = idx
            # Order colors by most constrained? No, just 0,1,2
            for c in range(3):
                conflict = False
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    mask -= lsb
                
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            
            return False
        
        return backtrack(0)

    best_adj = adj[:]
    best_count = edge_count(best_adj)
    
    # Local Search
    # We try to remove edges and add others to increase count
    # Since adding is constrained by triangle-free, we look for "flips"
    
    # Precompute non-edges for fast access?
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each edge
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Temporarily remove (u,v)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    # Try to add all possible edges greedily
                    # We need to find edges (a,b) not in adj such that N(a) & N(b) == 0
                    # And we must respect the current adj state
                    
                    # To be efficient, we only try to add edges that were previously blocked by (u,v)?
                    # Or just scan all non-edges? 120 edges is small.
                    
                    added = 0
                    new_adj = adj[:]
                    
                    # Sort potential edges? Random or fixed?
                    # Fixed order for determinism in this step, or random for exploration?
                    # Let's use a fixed order based on index to be fast.
                    for a in range(n):
                        for b in range(a + 1, n):
                            if not (new_adj[a] & (1 << b)):
                                # Check if adding (a,b) creates a triangle
                                if (new_adj[a] & new_adj[b]) == 0:
                                    new_adj[a] |= (1 << b)
                                    new_adj[b] |= (1 << a)
                                    added += 1
                                    
                    new_count = edge_count(new_adj)
                    
                    # Restore adj
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    if new_count > best_count:
                        # Check 3-colorability
                        if not is_3_colorable(new_adj):
                            best_adj = new_adj
                            best_count = new_count
                            improved = True
                            # Restart search from new best
                            break
                            
            if improved:
                break
                
    return get_edges_from_adj(best_adj)
