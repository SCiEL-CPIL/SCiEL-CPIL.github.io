"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on vertices 0..11 consisting of two disjoint K_{3,3} components.
    
    K_{3,3} is a bipartite graph with partitions of size 3.
    It has minimum degree 3.
    It is triangle-free (girth 4).
    The simple cycles in K_{3,3} have lengths 4 and 6 only.
    Thus, the set of distinct cycle lengths is {4, 6}, and the count is 2.
    """
    edges = []
    
    # Component 1: Vertices 0,1,2 (Part A) and 3,4,5 (Part B)
    part_a_1 = [0, 1, 2]
    part_b_1 = [3, 4, 5]
    for u in part_a_1:
        for v in part_b_1:
            edges.append([u, v])
            
    # Component 2: Vertices 6,7,8 (Part A) and 9,10,11 (Part B)
    part_a_2 = [6, 7, 8]
    part_b_2 = [9, 10, 11]
    for u in part_a_2:
        for v in part_b_2:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
