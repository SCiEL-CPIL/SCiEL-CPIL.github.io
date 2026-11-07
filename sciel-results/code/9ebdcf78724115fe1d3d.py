"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 16
    # The optimal triangle-free graph on n vertices is the complete bipartite graph
    # with parts as equal as possible. For n=16, this is K_{8,8}.
    # This graph has 64 edges.
    # It has diameter 2.
    # Removing any edge (u,v) in a triangle-free graph removes the only path of length 1.
    # Since there are no triangles, there are no alternative paths of length 2 between u and v.
    # Thus the distance becomes >= 3 or infinite, satisfying the essential edge condition.
    
    # Partition vertices into two sets of size 8
    set_a = set(range(8))
    set_b = set(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            # Ensure u < v for consistency, though bipartite sets are disjoint
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    # Convert to Python ints explicitly
    return [[int(e[0]), int(e[1])] for e in edges]
# EVOLVE-BLOCK-END
