# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 16
    # The maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4),
    # achieved by the complete bipartite graph K_{n/2, n/2}.
    # For n=16, K_{8,8} has 64 edges and diameter 2.
    # K_{8,8} is edge-critical: deleting any edge increases the diameter to 3.
    
    set_a = range(8)
    set_b = range(8, 16)
    
    edges = [[u, v] for u, v in itertools.product(set_a, set_b)]
    
    return edges
# EVOLVE-BLOCK-END
