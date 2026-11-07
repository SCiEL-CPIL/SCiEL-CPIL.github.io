"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 16}

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
    This graph is triangle-free, connected, and has diameter 2.
    Since it is triangle-free, every edge is critical (removal increases distance between endpoints to 3).
    Number of edges: 8 * 8 = 64.
    """
    n = P['n']
    # Split vertices into two sets of size n/2
    mid = n // 2
    set_a = list(range(mid))
    set_b = list(range(mid, n))
    
    edges = []
    # Connect every vertex in A to every vertex in B
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
