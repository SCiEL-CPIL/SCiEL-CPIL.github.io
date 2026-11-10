"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph consisting of two disjoint copies of K_{3,3}.
    K_{3,3} is a 3-regular bipartite graph with girth 4.
    The simple cycles in K_{3,3} have lengths 4 and 6 only.
    Thus, the union of two disjoint K_{3,3} has distinct cycle lengths {4, 6}.
    Number of distinct cycle lengths = 2.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
