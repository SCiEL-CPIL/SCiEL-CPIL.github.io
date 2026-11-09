"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct the minimal 3-uniform 3-chromatic hypergraph on 9 vertices.
    
    The Fano plane is the unique 3-uniform 3-chromatic hypergraph on 7 vertices.
    It has 7 edges. It is not 2-colorable (does not have Property B).
    We use vertices 0-6 for the Fano plane, and vertices 7-8 are isolated.
    
    The raw value is the number of edges, which is 7.
    This is the theoretical minimum for 3-uniform 3-chromatic hypergraphs.
    """
    
    # Fano plane edges on vertices 0..6
    # Standard representation of the 7 lines of the Fano plane:
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
    edges = [[int(x) for x in edge] for edge in edges_raw]
    
    # Valid 3-coloring for the Fano plane vertices (0-6)
    # Let's verify the coloring:
    # c(0)=0, c(1)=1, c(2)=2, c(3)=1, c(4)=2, c(5)=0, c(6)=1
    # Edge [0,1,2]: 0,1,2 -> Non-mono
    # Edge [0,3,4]: 0,1,2 -> Non-mono
    # Edge [0,5,6]: 0,0,1 -> Non-mono
    # Edge [1,3,5]: 1,1,0 -> Non-mono
    # Edge [1,4,6]: 1,2,1 -> Non-mono
    # Edge [2,3,6]: 2,1,1 -> Non-mono
    # Edge [2,4,5]: 2,2,0 -> Non-mono
    
    # Isolated vertices 7, 8 can be any color, e.g., 0.
    colors_raw = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are Python int
    colors = [int(c) for c in colors_raw]
    
    return {
        'edges': edges,
        'colors': colors
    }

# EVOLVE-BLOCK-END
