import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill
    
    def _creates_c4(adj_bitmask, u, v, n):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        # Get common neighbors of u and v's neighbors
        # C4 exists if there's x in N(u) and y in N(v) such that x-y is edge
        # Equivalent: (N(u) & N(v)) has an edge? No.
        # C4: u-x-y-v where x in N(u), y in N(v), x-y edge.
        # So: for some x in N(u), is there y in N(v) with x-y edge?
        # i.e., N(x) & N(v) != 0 for some x in N(u)
        nu = adj_bitmask[u]
        nv = adj_bitmask[v]
        # Iterate over smaller set
        if nu.bit_count() > nv.bit_count():
            nu, nv = nv, nu
        x = nu
        while x:
            low = x & (-x)
            xi = low.bit_length() - 1
            if adj_bitmask[xi] & nv:
                return True
            x ^= low
        return False

    def _try_construct(n, min_deg, time_limit):
        start_time = time.time()
        best_result = None
        
        for attempt in range(20000):
            if time.time() - start_time > time_limit:
                break
                
            # Use bitmask adjacency: adj[i] is an integer bitmask of neighbors
            adj = [0] * n
            
            success = False
            max_steps = n * n * 20
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check if min degree satisfied
                min_deg_curr = min(adj[i].bit_count() for i in range(n))
                if min_deg_curr >= min_deg:
                    success = True
                    break
                    
                # Find vertex with min degree
                u = min(range(n), key=lambda i: adj[i].bit_count())
                
                # Find candidates: vertices not adjacent to u
                candidates = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
                random.shuffle(candidates)
                
                added = False
                for v in candidates:
                    if not _creates_c4(adj, u, v, n):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
                        
                if not added:
                    # Stuck: try a repair - remove a random edge to break a C4
                    # Find a C4 and remove one edge
                    c4_found = False
                    for _ in range(50):  # Try a few random pairs
                        a = random.randint(0, n - 1)
                        b = random.randint(0, n - 1)
                        if a == b or not (adj[a] >> b) & 1:
                            continue
                        # Check if (a,b) is part of a C4
                        # Remove it and see if we're still C4-free? No, we need to check if removing helps.
                        # Simpler: just remove a random edge incident to u
                        if adj[u]:
                            x = adj[u]
                            low = x & (-x)
                            xi = low.bit_length() - 1
                            adj[u] ^= (1 << xi)
                            adj[xi] ^= (1 << u)
                            c4_found = True
                            break
                    if not c4_found:
                        break
                    
            if success:
                edges = []
                for u in range(n):
                    x = adj[u]
                    while x:
                        low = x & (-x)
                        v = low.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        x ^= low
                return {'n': n, 'edges': edges}
                
        return None

    # Try n=47 first (if possible)
    res = _try_construct(47, 8, 100.0)
    if res:
        return res

    # Try n=46
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        res = _try_construct(46, 7, remaining)
        if res:
            return res

    # Try n=45
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        res = _try_construct(45, 6, remaining)
        if res:
            return res

    # Fallback
    return {'n': 1, 'edges': []}
