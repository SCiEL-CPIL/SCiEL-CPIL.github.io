"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
    This graph is triangle-free, has diameter 2, and is edge-critical
    for diameter 2 (removing any edge increases diameter to 3).
    Number of edges: 64.
    """
    n = 16
    # Partition vertices into two sets of size 8
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    # Ensure types are strict Python ints and lists
    # The loop above produces ints and lists of ints.
    
    return edges
# EVOLVE-BLOCK-END
