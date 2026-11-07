"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    - Vertices 0-5 form K_{3,3} with partitions {0,1,2} and {3,4,5}.
    - Vertices 6-11 form K_{3,3} with partitions {6,7,8} and {9,10,11}.
    
    Properties:
    - Simple, undirected.
    - Minimum degree is 3.
    - Bipartite, so triangle-free (girth >= 4).
    - Contains cycles of length 4 and 6.
    - No other cycle lengths are possible in K_{3,3}.
    - Distinct cycle lengths: {4, 6}. Count = 2.
    """
    edges = []
    
    # First K_{3,3} component: vertices 0-5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Second K_{3,3} component: vertices 6-11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    # u and v are already ints from the lists above.
    # edges is a list of lists of ints.
    
    return edges

# EVOLVE-BLOCK-END
