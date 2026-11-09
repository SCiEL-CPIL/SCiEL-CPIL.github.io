import time

def construct():
    """
    Returns a dict with keys 'edges' and 'colors'.
    'edges': list of distinct triples (lists of 3 ints) on vertices 0..8.
    'colors': list of 9 ints from 0..2.
    Every edge must be non-monochromatic in the provided coloring.
    The hypergraph must be 3-chromatic (not 2-colorable).
    """
    start_time = time.time()
    deadline = start_time + 58.0 # Safety margin before 60s limit
    
    # The Fano plane is the smallest 3-uniform 3-chromatic hypergraph.
    # It has exactly 7 edges. This is the theoretical minimum.
    # We construct it on vertices 0..6 using the cyclic shift of {0, 1, 3} mod 7.
    # Vertices 7 and 8 are isolated.
    
    # Generate Fano plane edges via cyclic shifts
    # Base set: {0, 1, 3}
    # Shift i: {i, i+1, i+3} mod 7
    fano_edges = set()
    for i in range(7):
        v0 = i % 7
        v1 = (i + 1) % 7
        v2 = (i + 3) % 7
        edge = tuple(sorted((v0, v1, v2)))
        fano_edges.add(edge)
        
    # Convert to list of lists, sorted for determinism
    edges = [list(e) for e in sorted(fano_edges)]
    
    # Define a valid 3-coloring for the Fano plane vertices (0..6)
    # We need to ensure no edge is monochromatic.
    # Let's use the coloring: 0, 1, 2, 1, 2, 0, 1
    # Let's verify:
    # Edges are:
    # 0: {0,1,3} -> colors 0,1,1 (OK)
    # 1: {1,2,4} -> colors 1,2,2 (OK)
    # 2: {2,3,5} -> colors 2,1,0 (OK)
    # 3: {3,4,6} -> colors 1,2,1 (OK)
    # 4: {0,4,5} -> colors 0,2,0 (OK)  (Note: {4,5,0} sorted is {0,4,5})
    # 5: {1,5,6} -> colors 1,0,1 (OK)  (Note: {5,6,1} sorted is {1,5,6})
    # 6: {0,2,6} -> colors 0,2,1 (OK)  (Note: {6,0,2} sorted is {0,2,6})
    
    colors_0_6 = [0, 1, 2, 1, 2, 0, 1]
    
    # Vertices 7 and 8 are isolated, assign color 0
    colors_7_8 = [0, 0]
    
    colors = colors_0_6 + colors_7_8
    
    # Final Type Conversion and Validation
    # Ensure all elements are Python int
    edges_out = []
    for e in edges:
        edges_out.append([int(e[0]), int(e[1]), int(e[2])])
        
    colors_out = [int(c) for c in colors]
    
    # Sanity Check: No monochromatic edges
    for e in edges_out:
        if colors_out[e[0]] == colors_out[e[1]] == colors_out[e[2]]:
            raise ValueError("Monochromatic edge detected in Fano plane coloring")

    if time.time() > deadline:
        # Fallback, though this should never happen for this static construction
        pass

    return {
        'edges': edges_out,
        'colors': colors_out
    }
