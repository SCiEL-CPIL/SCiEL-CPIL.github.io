"""
Constructs a 3-uniform hypergraph on 9 vertices (0-8) that is not 2-colorable
with the minimum number of edges.
The Fano plane is the unique minimal 3-uniform non-2-colorable hypergraph,
having 7 edges. Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
This static solution achieves the theoretical lower bound for edge count.
"""

def construct():
    # Fano plane edges on vertices 0-6
    # This is a 3-uniform hypergraph with chromatic number 3.
    # It is the unique (up to isomorphism) minimal 3-uniform hypergraph
    # that is not 2-colorable (violates Property B).
    fano_edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )
    
    # A valid 3-coloring for the Fano plane.
    # Vertices 7 and 8 are isolated, so their colors can be anything.
    # We assign them 0 for simplicity.
    fano_colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)
    
    # Convert to lists of Python integers to satisfy type constraints
    # Ensure strict int types to avoid numpy/bool issues
    edges = [tuple(int(v) for v in edge) for edge in fano_edges]
    colors = [int(c) for c in fano_colors]
    
    return {
        'edges': edges,
        'colors': colors
    }
