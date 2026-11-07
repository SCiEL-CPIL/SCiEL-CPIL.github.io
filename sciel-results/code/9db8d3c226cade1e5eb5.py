import time

def construct():
    # Deadline guard (not strictly necessary for O(1) construction, but required by directive)
    start_time = time.time()
    deadline = start_time + 59.0  # 59s safety margin
    
    # The Fano plane is the minimal 3-uniform hypergraph that is not 2-colorable.
    # It has 7 edges. We embed it in 9 vertices by adding isolated vertices 7 and 8.
    # This configuration is 3-chromatic (requires 3 colors, not 2).
    
    # Verified valid edges for the Fano plane on vertices 0-6
    # This specific labeling is known to work with the coloring below.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # 3-coloring for vertices 0-8
    # Vertices 7 and 8 are isolated, so their color is arbitrary (set to 0).
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are exactly Python int
    # The problem states: "integers must be Python int (NumPy integers and bools are rejected)"
    # Converting explicitly to be safe.
    final_edges = []
    for e in edges:
        final_edges.append([int(v) for v in e])
        
    final_colors = [int(c) for c in colors]
    
    # Check if we are still within time limit (should be, as this is instant)
    if time.time() > deadline:
        # Fallback to a known valid result if somehow delayed (unlikely)
        pass 
        
    return {
        'edges': final_edges,
        'colors': final_colors
    }
