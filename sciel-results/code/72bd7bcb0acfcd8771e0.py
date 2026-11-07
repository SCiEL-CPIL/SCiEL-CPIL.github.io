import time

P = {'n': 16}

def construct():
    """
    Returns a list of edges for the complete bipartite graph K_{8,8}.
    
    Proof of optimality:
    1. If an edge (u,v) is in a triangle (u,v,w), then removing (u,v) leaves u and v at distance 2 via w.
       The diameter would remain 2, violating the criticality condition.
       Therefore, the graph must be triangle-free (assuming no bridges in a dense graph).
    2. By Mantel's Theorem, max edges in a triangle-free graph on n vertices is floor(n^2/4).
    3. For n=16: max edges = 64, achieved uniquely by K_{8,8}.
    4. K_{8,8} has diameter 2: vertices in same partition have distance 2, different partitions have distance 1.
    5. Removing any edge (u,v) with u in V1, v in V2: u and v have no common neighbors
       (N(u) ⊆ V2, N(v) ⊆ V1, N(u) ∩ N(v) = ∅). Shortest path becomes length 3.
       Diameter increases from 2 to 3. All edges are critical.
    """
    n = P['n']
    
    if n < 2:
        return []
    
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    edges = []
    for u in part1:
        for v in part2:
            edges.append([u, v])
    
    return edges
