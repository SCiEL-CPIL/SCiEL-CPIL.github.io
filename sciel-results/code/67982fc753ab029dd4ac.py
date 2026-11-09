"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a C5-free graph on 32 vertices with independence number 8.
    The graph is 8 disjoint copies of K4.
    This is a known optimal construction for this Ramsey number context.
    """
    n = 32
    edges = []
    for b in range(8):
        offset = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([offset + i, offset + j])
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
