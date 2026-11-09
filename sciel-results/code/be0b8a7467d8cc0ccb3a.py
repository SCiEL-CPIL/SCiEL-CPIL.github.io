"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the Fano plane (PG(2,2)) as a 3-uniform hypergraph on 9 vertices.
    The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable
    (i.e., it is 3-chromatic). It has 7 edges.
    Vertices 7 and 8 are isolated, satisfying the 9-vertex constraint.
    The number of edges (7) is the theoretical minimum for a 3-chromatic
    3-uniform hypergraph.
    
    This implementation hardcodes the edges and a valid 3-coloring to ensure
    maximum robustness and avoid any potential generation errors.
    """
    # The Fano plane edges on vertices 0..6
    # These are the 7 lines of the Fano plane
    edges = [
        [0, 1, 3],
        [0, 2, 4],
        [0, 5, 6],
        [1, 2, 5],
        [1, 4, 6],
        [2, 3, 6],
        [3, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # Colors for 7 and 8 can be anything, we use 0.
    # Verified that no edge is monochromatic:
    # (0,1,3) -> (0,1,0) OK
    # (0,2,4) -> (0,2,0) OK
    # (0,5,6) -> (0,1,2) OK
    # (1,2,5) -> (1,2,1) OK
    # (1,4,6) -> (1,0,2) OK
    # (2,3,6) -> (2,0,2) OK
    # (3,4,5) -> (0,0,1) OK
    colors = [0, 1, 2, 0, 0, 1, 2, 0, 0]
    
    # Ensure types are Python int
    return {
        'edges': [[int(e[0]), int(e[1]), int(e[2])] for e in edges],
        'colors': [int(c) for c in colors]
    }

# EVOLVE-BLOCK-END
