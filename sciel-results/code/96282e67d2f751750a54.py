"""
Rewritten program using a deterministic optimal solution (Fano Plane).

The Fano plane is the unique 3-uniform hypergraph on 7 vertices that is 
3-chromatic and minimal. It has 7 edges. 
It is a proven result that any 3-uniform hypergraph with fewer than 7 edges 
is 2-colorable. Thus, 7 is the global minimum for the number of edges in 
a 3-chromatic 3-uniform hypergraph.

The task specifies vertices 0..8. We use the Fano plane on vertices 0..6 
and leave vertices 7 and 8 isolated. Isolated vertices do not affect the 
chromatic number or the edge count.
"""

def construct():
    # Fano plane edges on vertices 0-6
    # The Fano plane is the projective plane of order 2.
    # Vertices: 0, 1, 2, 3, 4, 5, 6
    # Edges are lines of the plane.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane.
    # The Fano plane is not 2-colorable, so it requires 3 colors.
    # Coloring:
    # v0: 0
    # v1: 1
    # v2: 2
    # v3: 1
    # v4: 2
    # v5: 0
    # v6: 1
    # Let's verify non-monochromaticity:
    # [0,1,2] -> 0,1,2 (OK)
    # [0,3,4] -> 0,1,2 (OK)
    # [0,5,6] -> 0,0,1 (OK)
    # [1,3,5] -> 1,1,0 (OK)
    # [1,4,6] -> 1,2,1 (OK)
    # [2,3,6] -> 2,1,1 (OK)
    # [2,4,5] -> 2,2,0 (OK)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strict Python int and list
    edges_out = [[int(x) for x in e] for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}
