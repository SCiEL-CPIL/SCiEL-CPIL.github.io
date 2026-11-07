"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _is_two_colorable(edges):
    """Check if the hypergraph with given edges admits a 2-coloring (no monochromatic edge)."""
    n = 9
    # For each of the 2^n colorings, check if any edge is monochromatic
    # Use bitmasks for speed
    for mask in range(1 << n):
        valid = True
        for e in edges:
            # Check if all three vertices have the same color
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True
    return False

def _find_three_coloring(edges):
    """Find a 3-coloring where no edge is monochromatic. Returns list of 9 colors or None."""
    n = 9
    # Try all 3^9 = 19683 colorings
    for assignment in range(3 ** n):
        colors = []
        temp = assignment
        valid = True
        for i in range(n):
            colors.append(temp % 3)
            temp //= 3
        # Check all edges
        for e in edges:
            if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                valid = False
                break
        if valid:
            return colors
    return None

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # The Fano plane has 7 edges on 7 vertices and is 3-chromatic (not 2-colorable)
    # Fano plane edges (on vertices 0-6):
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # We need to add edges involving vertices 7 and 8 to make the whole thing 3-chromatic
    # The key insight: we need the hypergraph on 9 vertices to NOT be 2-colorable
    # 
    # Strategy: Start with Fano plane on {0..6}, then add edges that force vertices 7 and 8
    # to be "trapped" such that no 2-coloring works.
    #
    # Let's think about this more carefully. We want minimum edges.
    # The Fano plane alone (7 edges) on 7 vertices is not 2-colorable.
    # But we have 9 vertices. The question is whether adding 2 isolated vertices keeps it non-2-colorable.
    # Actually, if we just have the Fano plane edges and vertices 7,8 are isolated, 
    # then we can 2-color it: color vertices 7,8 arbitrarily, and the Fano part can't be 2-colored.
    # Wait - the Fano plane IS not 2-colorable, so any hypergraph containing it as a subgraph is not 2-colorable.
    # So 7 edges (just the Fano plane) should suffice!
    
    # Let's verify: is the Fano plane 3-chromatic? Yes, it's a well-known fact.
    # The chromatic number of the Fano plane (as a 3-uniform hypergraph) is 3.
    
    # But wait - we need exactly 3-chromatic, meaning it requires exactly 3 colors.
    # The Fano plane requires 3 colors (not 2, and not 1).
    
    # So with 7 edges (Fano plane on vertices 0-6, vertices 7 and 8 unused), 
    # the hypergraph is 3-chromatic.
    
    # However, we need to provide a valid 3-coloring. Let's find one.
    # Fano plane 3-coloring: vertices 0-6 need 3 colors, no line monochromatic.
    # One known 3-coloring: {0,1,2} -> 0, {3,4,5} -> 1, {6} -> ... 
    # Actually let's just search for it.
    
    edges = [list(e) for e in fano_edges]
    
    # Find a valid 3-coloring
    colors = _find_three_coloring(edges)
    
    if colors is None:
        # This shouldn't happen, but fallback
        colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify it's not 2-colorable
    if _is_two_colorable(edges):
        # This should not happen for Fano plane, but if it does, we need more edges
        # Add edges to force non-2-colorability
        # Add edge (0, 7, 8) and others
        extra_edges = [(0, 7, 8), (1, 7, 8), (2, 7, 8)]
        for e in extra_edges:
            edges.append(list(e))
        
        colors = _find_three_coloring(edges)
        if colors is None:
            colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Shouldn't happen for this fixed construction, but safety
        pass
    
    # Ensure types are correct
    edges = [list(e) for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
