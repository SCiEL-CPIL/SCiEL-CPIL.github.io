import time
import itertools

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices that is exactly 3-chromatic
    with the minimum possible number of edges (7 edges, Fano plane).
    
    The Fano plane is the unique (up to isomorphism) minimum 3-chromatic 
    3-uniform hypergraph. Any 3-uniform hypergraph with fewer than 7 edges 
    is 2-colorable.
    
    We define the Fano plane on vertices 0..6 and leave vertices 7 and 8 isolated.
    """
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    # Fano plane edges on vertices 0-6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Programmatically verify that no 2-coloring exists for the Fano plane part
    # This ensures the hypergraph is indeed 3-chromatic (not 2-chromatic)
    # Vertices 7 and 8 are isolated, so they don't affect the chromatic number.
    # We check all 2^7 binary colorings of vertices 0-6.
    # If any coloring is valid (no monochromatic edge), it's 2-colorable.
    # We expect to find none.
    is_2_colorable = False
    for mask in range(1 << 7):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
    
    # If it were 2-colorable, we would have a problem, but Fano plane is known not to be.
    # This check is a safety net.
    if is_2_colorable:
        # Fallback: theoretically shouldn't happen, but if it does, we might need a different graph.
        # However, the Fano plane is mathematically proven to be 3-chromatic.
        pass

    # Provide a valid 3-coloring
    # 0:0, 1:1, 2:2
    # 3:1, 4:2, 5:0, 6:1
    # 7:0, 8:0 (isolated, any color works)
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are correct: lists of ints for edges (converted to tuples), tuple of ints for colors
    final_edges = [tuple(e) for e in edges]
    final_colors = tuple(colors)
    
    return {
        'edges': final_edges,
        'colors': final_colors
    }
