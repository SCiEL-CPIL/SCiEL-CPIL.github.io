def construct():
    # The Fano plane is the unique 3-chromatic 3-uniform hypergraph with 7 edges.
    # This is the global minimum for the number of edges in a 3-chromatic triple system.
    # Vertices 7 and 8 are isolated and do not affect the chromatic number.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring for the Fano plane
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Return with strict types
    return {
        'edges': [[int(x) for x in e] for e in edges],
        'colors': [int(c) for c in colors]
    }
