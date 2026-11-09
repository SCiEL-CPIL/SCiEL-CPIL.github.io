"""
Solution for: Few hyperedges forcing three colors.
Constructs a 3-uniform hypergraph on 9 vertices (0..8) with chromatic number 3.
Uses the Fano plane structure on vertices 0..6, which is the minimal such hypergraph (7 edges).
Vertices 7 and 8 are isolated.
"""

def construct():
    # Define the 7 edges of the Fano plane.
    # The Fano plane is a 2-(7,3,1) design, meaning every pair of vertices appears in exactly one edge.
    # It is known that the Fano plane is not 2-colorable (bipartite), so its chromatic number is 3.
    # Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    
    # Edge list as distinct triples
    h_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Assign colors 0, 1, 2 to vertices 0..6 such that no edge is monochromatic.
    # A valid 3-coloring for the Fano plane can be found by assigning colors based on 
    # a cyclic permutation or specific mapping. 
    # One valid mapping for vertices 0-6:
    # v0: 0, v1: 1, v2: 2
    # v3: 1, v4: 2, v5: 0
    # v6: 1
    # Let's verify edge (0,1,2): 0,1,2 -> ok
    # (0,3,4): 0,1,2 -> ok
    # (0,5,6): 0,0,1 -> ok
    # (1,3,5): 1,1,0 -> ok
    # (1,4,6): 1,2,1 -> ok
    # (2,3,6): 2,1,1 -> ok
    # (2,4,5): 2,2,0 -> ok
    
    # Vertices 7 and 8 are isolated, their color doesn't affect edge constraints.
    h_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure all values are Python ints
    for i in range(len(h_edges)):
        for j in range(3):
            h_edges[i][j] = int(h_edges[i][j])
            
    for i in range(len(h_colors)):
        h_colors[i] = int(h_colors[i])

    return {
        'edges': h_edges,
        'colors': h_colors
    }
