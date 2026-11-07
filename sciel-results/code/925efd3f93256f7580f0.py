import time

P = {'n': 16}

def construct():
    """
    Construct a diameter-2-critical graph on 16 vertices maximizing edge count.
    
    Theoretical analysis:
    For a graph G to be diameter-2-critical (every edge essential for diameter 2),
    removing any edge must increase the diameter to >2. This implies that for every
    edge (u,v), there is exactly one common neighbor of u and v in G. Otherwise:
    - If 0 common neighbors: d(u,v) becomes >2 after removal (good for criticality)
    - If >=2 common neighbors: d(u,v) remains 2 after removal (not critical)
    
    The maximum number of edges in such a graph on n vertices is floor(n^2/4),
    achieved by K_{n/2, n/2} for even n. For n=16, this is K_{8,8} with 64 edges.
    """
    n = P['n']
    mid = n // 2
    
    # Build complete bipartite graph K_{8,8}
    edges = []
    for u in range(mid):
        for v in range(mid, n):
            edges.append([u, v])
    
    return edges
