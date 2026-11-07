def construct():
    # The Fano plane is the unique minimal 3-chromatic 3-uniform hypergraph.
    # It has 7 vertices and 7 edges.
    # We map these to vertices 0-6 of the 9-vertex system.
    # Vertices 7 and 8 are isolated.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane.
    # Vertex colors: 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Vertices 7 and 8 are assigned color 0 (arbitrary, as they are isolated)
    
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are strictly Python int and list/tuple
    # Convert edges to list of lists of ints
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1]), int(e[2])])
        
    # Ensure colors are Python ints
    final_colors = [int(c) for c in colors]
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
