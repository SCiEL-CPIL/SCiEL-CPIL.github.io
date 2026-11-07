"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0-11) with minimum degree 3 and girth at least 4.
    The goal is to minimize the number of distinct simple cycle lengths.
    
    Theoretical Analysis:
    - Minimum degree >= 3 implies the graph cannot be a disjoint union of cycles (which have degree 2).
    - Girth >= 4 implies no triangles.
    - It is a known graph theory result that any graph with minimum degree >= 3 and girth >= 4 must contain cycles of at least two distinct lengths.
    - The graph consisting of two disjoint copies of K_{3,3} achieves a raw value of 2 (cycle lengths 4 and 6).
    - Since 1 is impossible, 2 is the minimum raw value.
    
    Construction:
    - Component 1: K_{3,3} on vertices {0, 1, 2} and {3, 4, 5}.
    - Component 2: K_{3,3} on vertices {6, 7, 8} and {9, 10, 11}.
    
    This graph is:
    - Simple and undirected.
    - Triangle-free (bipartite).
    - Minimum degree 3.
    - Contains cycles of length 4 and 6 only.
    """
    edges = []
    
    # Component 1: K_{3,3}
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3}
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Ensure types are strictly Python int and list
    # The loops above already produce Python ints and lists of ints.
    # No conversion needed, but we return the list directly.
    return edges

# EVOLVE-BLOCK-END
