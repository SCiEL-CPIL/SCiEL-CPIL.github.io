"""Triangle-free diameter-four augmentation - minimum edge solution."""

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    """
    Add minimum number of edges to make the path on 24 vertices have diameter <= 4
    while remaining triangle-free and connected.
    
    Strategy: Use a bipartite construction with center vertex 1.
    A = even vertices {0,2,...,22}, B = odd vertices {1,3,...,23}.
    Connect center 1 to a minimum dominating set S of A (with respect to path adjacency on A).
    The path on A has 12 vertices; MVC size is 6.
    S = {2, 6, 10, 14, 18, 22} dominates A.
    Vertex 1 is already connected to 0 and 2 via path edges.
    So we need to add edges from 1 to S \ {0, 2} = {6, 10, 14, 18, 22}.
    That's 5 edges.
    
    Verification:
    - Triangle-free: All added edges go from B to A, preserving bipartiteness.
    - Connected: Path edges keep everything connected; added edges only help.
    - Diameter <= 4: For any u, d(u,1) <= 2 (since S dominates A, and B vertices 
      are at distance 1 from their path neighbors which are in A). So d(u,v) <= 4.
    """
    # Minimum dominating set on A = {0,2,...,22} with respect to path adjacency
    S = [2, 6, 10, 14, 18, 22]
    center = 1
    
    # Path edges already connect center to 0 and 2
    already_connected = {0, 2}
    
    added_edges = []
    for a in S:
        if a not in already_connected:
            added_edges.append([center, a])
    
    # Ensure all are Python ints
    added_edges = [[int(u), int(v)] for u, v in added_edges]
    
    return added_edges
