"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_{3,3} components.
    - Component 1: Vertices 0..5. Partitions {0,1,2} and {3,4,5}.
    - Component 2: Vertices 6..11. Partitions {6,7,8} and {9,10,11}.
    
    Properties:
    - Simple undirected graph.
    - Minimum degree 3 (each vertex connected to 3 others).
    - Girth 4 (bipartite, smallest cycle is 4).
    - Contains cycles.
    - Distinct simple cycle lengths: 4 and 6. (Count = 2).
    """
    edges = set()
    
    # Component 1: K_{3,3} on vertices 0-5
    # Partitions A1 = {0, 1, 2}, B1 = {3, 4, 5}
    A1 = [0, 1, 2]
    B1 = [3, 4, 5]
    for u in A1:
        for v in B1:
            edges.add(tuple(sorted((u, v))))
            
    # Component 2: K_{3,3} on vertices 6-11
    # Partitions A2 = {6, 7, 8}, B2 = {9, 10, 11}
    A2 = [6, 7, 8]
    B2 = [9, 10, 11]
    for u in A2:
        for v in B2:
            edges.add(tuple(sorted((u, v))))
            
    return [list(e) for e in sorted(edges)]

# EVOLVE-BLOCK-END
