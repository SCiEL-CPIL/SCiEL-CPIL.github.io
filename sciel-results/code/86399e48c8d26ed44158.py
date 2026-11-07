"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    
    # Strategy: Complete Bipartite Graph K_{n/2, n/2}
    # This graph has diameter 2.
    # Removing any edge (u,v) where u in A, v in B increases dist(u,v) to 3.
    # Thus, every edge is essential for diameter 2.
    # Edge count is (n/2)^2. For n=16, 8*8=64.
    
    half = n // 2
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        if time.time() > deadline:
            break
        for v in B:
            if time.time() > deadline:
                break
            edges.append([u, v])
            
    # Ensure types are correct
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges
# EVOLVE-BLOCK-END
