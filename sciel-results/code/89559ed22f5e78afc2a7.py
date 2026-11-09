import time
import random
import numpy as np

N = 40
DEADLINE = 100  # Seconds

def _creates_c6(adj, u, v):
    """
    Checks if adding edge (u, v) creates a C6.
    This is true if there is a simple path of length 5 from u to v in the current graph.
    Path: u - a - b - c - d - v
    Vertices u, a, b, c, d, v must be distinct.
    """
    # Get neighbors of u and v
    # adj is a list of bitmasks
    nu = adj[u]
    nv = adj[v]
    
    # Iterate a in N(u)
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        if a == v: continue # a cannot be v (simple path, distinct vertices)
        
        # Iterate d in N(v)
        d_mask = nv
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            if d == u: continue
            if d == a: continue
            
            # We need a path a - b - c - d
            # b in N(a), c in N(b)
            # b, c must be distinct from u, v, a, d
            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            
            nb_mask = adj[a] & ~forbidden
            while nb_mask:
                lsb_b = nb_mask & -nb_mask
                b = lsb_b.bit_length() - 1
                nb_mask ^= lsb_b
                
                # c in N(b)
                nc_mask = adj[b] & ~forbidden
                while nc_mask:
                    lsb_c = nc_mask & -nc_mask
                    c = lsb_c.bit_length() - 1
                    nc_mask ^= lsb_c
                    
                    # Check if d is in N(c)
                    if (adj[c] >> d) & 1:
                        return True
                        
    return False

def construct():
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Run random restarts until deadline
    while time.time() - start_time < DEADLINE:
        random.shuffle(all_edges)
        adj = [0] * N
        count = 0
        
        for u, v in all_edges:
            if not _creates_c6(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        if count > best_count:
            best_count = count
            # Reconstruct edge list
            best_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
    
    return best_edges
