import time
import itertools

def construct():
    # The Fano plane is the smallest 3-chromatic 3-uniform hypergraph.
    # It has 7 edges and 7 vertices.
    # We embed it in 9 vertices (0..6 used, 7,8 isolated).
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring for the Fano plane
    colors = [0, 0, 1, 1, 2, 1, 2, 0, 0]
    
    return {
        'edges': [list(e) for e in edges],
        'colors': colors
    }
