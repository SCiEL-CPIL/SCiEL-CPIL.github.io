import time

def construct():
    deadline = time.time() + 45.0

    # The Fano plane on 7 vertices is the unique (up to isomorphism) smallest
    # 3-uniform hypergraph with chromatic number 3. It has exactly 7 edges.
    # A well-known result states that any 3-uniform hypergraph with fewer than
    # 7 edges is 2-colorable. Thus 7 is the global minimum for this problem.

    # We embed the Fano plane on vertices 0..6 and leave 7,8 isolated.
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
    # No edge is monochromatic, and at least 3 colors are required globally.
    colors = [0, 1, 2, 1, 1, 2, 0, 0, 0]

    # Ensure all integers are Python int (not numpy, not bool)
    final_edges = []
    for e in edges:
        final_edges.append([int(v) for v in e])
        
    final_colors = [int(c) for c in colors]

    return {'edges': final_edges, 'colors': final_colors}
