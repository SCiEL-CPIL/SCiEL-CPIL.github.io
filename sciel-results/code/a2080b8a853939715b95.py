"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the Fano plane (on vertices 0-6) with two isolated vertices (7-8).
    This is a 3-uniform hypergraph with 7 edges, which is the theoretical minimum
    for a 3-chromatic 3-uniform hypergraph.
    
    Robustness:
    - Explicitly uses Python int types.
    - Returns lists/tuples as required.
    - Includes a self-check for 2-colorability to ensure the hypergraph is 
      indeed 3-chromatic (i.e., not 2-colorable).
    """
    # Fano plane edges
    raw_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Ensure types are Python int
    edges = [tuple(int(v) for v in e) for e in raw_edges]
    
    # Colors for 9 vertices (0..8)
    # V0:0, V1:1, V2:2, V3:1, V4:2, V5:0, V6:1, V7:0, V8:0
    raw_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    colors = [int(c) for c in raw_colors]
    
    # Self-verification (cheap and fast)
    # 1. Check distinct edges
    assert len(set(map(tuple, edges))) == len(edges), "Edges must be distinct"
    
    # 2. Check non-monochromatic
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError(f"Edge {e} is monochromatic in provided coloring")
            
    # 3. Check 3-chromatic (i.e., NOT 2-colorable)
    # A hypergraph is 2-colorable if there is a partition of vertices into two sets 
    # such that no edge is contained in either set.
    # For the Fano plane, we know it is not 2-colorable. 
    # We can verify this by checking all 2^7 assignments of the 7 active vertices? 
    # 2^7 = 128 checks. Very fast.
    # If any assignment 2-colors it, we have a bug in our edge list.
    import itertools
    is_2_colorable = False
    for assignment in itertools.product([0, 1], repeat=7):
        # Map assignment to vertices 0-6
        # Check if any edge is monochromatic in this 2-coloring
        valid = True
        for e in edges:
            c0, c1, c2 = assignment[e[0]], assignment[e[1]], assignment[e[2]]
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        raise ValueError("Constructed hypergraph is 2-colorable, but must be 3-chromatic")

    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
