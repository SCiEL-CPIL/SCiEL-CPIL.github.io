"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    
    # The problem asks for a graph with diameter 2 where every edge is critical.
    # This implies the graph must be triangle-free (if a triangle exists, 
    # removing an edge from it leaves a path of length 2, so diameter stays 2).
    # By Mantel's Theorem, the maximum number of edges in a triangle-free 
    # graph on n vertices is floor(n^2 / 4).
    # For n=16, max edges = 64.
    # The complete bipartite graph K_{8,8} achieves this bound and satisfies 
    # the criticality condition: removing any edge (u,v) with u in A, v in B 
    # makes dist(u,v) = 3 (via u->b'->a'->v), increasing diameter to 3.
    
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
            
    return edges
# EVOLVE-BLOCK-END
