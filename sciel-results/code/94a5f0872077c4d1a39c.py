"""Rewritten program for the suite task; constructs two disjoint K3,3 components."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    The graph consists of two disjoint copies of the complete bipartite graph K3,3.
    - Vertices 0, 1, 2 form one part of the first K3,3.
    - Vertices 3, 4, 5 form the other part of the first K3,3.
    - Vertices 6, 7, 8 form one part of the second K3,3.
    - Vertices 9, 10, 11 form the other part of the second K3,3.
    
    Properties:
    - Every vertex has degree 3.
    - The graph is bipartite, so it is triangle-free (girth >= 4).
    - The only simple cycle lengths in K3,3 are 4 and 6.
    - Thus, the set of distinct simple cycle lengths is {4, 6}, which has size 2.
    """
    edges = []
    
    # First K3,3 component: vertices 0-5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Second K3,3 component: vertices 6-11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    # Return edges as a list of lists of integers
    return edges

# EVOLVE-BLOCK-END
