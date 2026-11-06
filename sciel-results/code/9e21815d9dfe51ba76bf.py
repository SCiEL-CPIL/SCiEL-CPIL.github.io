import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    n = 40
    
    # Bitset adjacency for fast operations
    adj = [0] * n
    
    def bit_count(x):
        return bin(x).count('1')
    
    def neighbors(u):
        return [i for i in range(n) if (adj[u] >> i) & 1]
    
    def would_create_c6(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A new C6 containing (u,v) would be u-w-z-y-x-v-u,
        which requires a path of length 5 from u to v in the current graph.
        Path: u - w - z - y - x - v
        w in N(u), x in N(v).
        Need path w-z-y-x of length 3.
        """
        Nu = neighbors(u)
        Nv = neighbors(v)
        
        # To minimize work, iterate over the smaller neighbor set first
        if len(Nu) > len(Nv):
            Nu, Nv = Nv, Nu
            u, v = v, u
        
        for w in Nu:
            for x in Nv:
                if w == x:
                    continue
                # Check if there's a path of length 3 from w to x
                # i.e., exists z such that w~z and z~y and y~x for some y
                # Simpler: for each z in N(w), check if N(z) intersects N(x)
                # But we must ensure the path is simple (no repeated vertices)
                # w, z, y, x must be distinct.
                
                for z in neighbors(w):
                    if z == x:
                        continue
                    # Check common neighbors of z and x
                    # We need a common neighbor y such that y != w
                    # y cannot be x because x is not in N(x) (no self-loops)
                    common = adj[z] & adj[x]
                    if common:
                        # Need a common neighbor y != w
                        if common & ~(1 << w):
                            return True
        return False
    
    # Initial graph: 8 disjoint K5s
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    best_edges = edges[:]
    best_count = len(edges)
    
    # Try multiple random restarts with different orders
    for restart in range(200):  # More restarts with faster C6 check
        if time.time() > deadline:
            break
        
        # Restore initial state
        adj = [0] * n
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        
        # Candidate edges: all non-edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    candidates.append((u, v))
        
        # Shuffle candidates to try different orders
        random.shuffle(candidates)
        
        # Greedy addition - use a set for faster "is non-edge" check
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Check if still a non-edge
            if not ((adj[u] >> v) & 1):
                # Check if adding this edge would create a C6
                if not would_create_c6(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    return best_edges
