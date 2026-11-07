"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a complete bipartite graph K_{8,8} on vertices 0..15.
    
    Reasoning:
    1. The condition "Deleting EACH individual edge must increase its diameter above two 
       or disconnect it" for a graph of diameter 2 implies the graph must be triangle-free.
       If a triangle existed, removing one edge would leave a path of length 2, keeping 
       the diameter at 2 (or less, but it's 2).
    2. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on 
       n=16 vertices is floor(16^2 / 4) = 64.
    3. The unique graph achieving this maximum is the complete bipartite graph K_{8,8}.
    4. K_{8,8} has diameter 2.
    5. Therefore, K_{8,8} is the optimal solution with 64 edges.
    """
    n = 16
    # Partition vertices into two sets of size 8
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        for v in set_b:
            # Ensure types are strict Python ints
            edges.append([int(u), int(v)])
            
    return edges
# EVOLVE-BLOCK-END
