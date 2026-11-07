"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_3,3 components.
    Each K_3,3 has minimum degree 3 and girth 4.
    The cycle lengths in K_3,3 are exactly 4 and 6.
    Thus, the whole graph has distinct cycle lengths {4, 6}, count = 2.
    
    It is a known result that any graph with minimum degree at least 3 must contain 
    cycles of at least two distinct lengths. Therefore, a count of 2 is minimal.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    edges = []
    
    # Component 1: K_3,3 on vertices 0, 1, 2 (left) and 3, 4, 5 (right)
    part1_left = [0, 1, 2]
    part1_right = [3, 4, 5]
    for u in part1_left:
        for v in part1_right:
            edges.append([int(u), int(v)])
            
    # Check deadline (unlikely to be needed for static construction)
    if time.time() > deadline:
        return edges

    # Component 2: K_3,3 on vertices 6, 7, 8 (left) and 9, 10, 11 (right)
    part2_left = [6, 7, 8]
    part2_right = [9, 10, 11]
    for u in part2_left:
        for v in part2_right:
            edges.append([int(u), int(v)])
            
    # Ensure edges are lists of Python ints
    # They already are.
    # No loops or duplicates by construction of K_3,3.
    # The graph is simple and undirected.
    # Min degree is 3.
    # Girth is 4 (bipartite, no triangles).
    # Cycles in K_3,3 are only of length 4 and 6.
    # Total distinct cycle lengths: 2.
    
    return edges

# EVOLVE-BLOCK-END
