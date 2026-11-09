"""Constructs the minimal 3-uniform 3-chromatic hypergraph on 9 vertices."""

def construct():
    """
    Returns a dict with keys 'edges' and 'colors'.
    
    The Fano plane is the unique 3-uniform 3-chromatic hypergraph on 7 vertices,
    with exactly 7 edges. This is the proven theoretical minimum for any
    3-chromatic 3-uniform hypergraph. Vertices 0-6 form the Fano plane,
    and vertices 7-8 are isolated.
    
    The raw value (number of edges) is 7, which is optimal.
    """
    
    # Fano plane edges on vertices 0..6
    # Standard representation of the 7 lines of the Fano plane:
    # Using the representation where lines are:
    # {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    
    edges_raw = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Convert to lists of Python ints to satisfy type checking requirements
    # Ensure all elements are native Python int (not numpy int, not bool)
    edges = [[int(x) for x in edge] for edge in edges_raw]
    
    # Valid 3-coloring for the Fano plane vertices (0-6)
    # Assignment:
    # c(0)=0, c(1)=1, c(2)=2
    # c(3)=1, c(4)=2, c(5)=0
    # c(6)=1
    #
    # Verification that no edge is monochromatic:
    # Edge [0,1,2]: colors 0,1,2 -> distinct, non-mono
    # Edge [0,3,4]: colors 0,1,2 -> distinct, non-mono
    # Edge [0,5,6]: colors 0,0,1 -> not all same, non-mono
    # Edge [1,3,5]: colors 1,1,0 -> not all same, non-mono
    # Edge [1,4,6]: colors 1,2,1 -> not all same, non-mono
    # Edge [2,3,6]: colors 2,1,1 -> not all same, non-mono
    # Edge [2,4,5]: colors 2,2,0 -> not all same, non-mono
    #
    # Isolated vertices 7, 8 can be assigned any color, e.g., 0.
    
    colors_raw = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are Python int
    colors = [int(c) for c in colors_raw]
    
    return {
        'edges': edges,
        'colors': colors
    }
