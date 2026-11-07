"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_2_colorable(edges, n=9):
    """
    Check if a 3-uniform hypergraph is 2-colorable.
    Uses backtracking with pruning.
    """
    if not edges:
        return True
    
    # Simple memoization or just recursive backtracking
    # For n=9, there are at most 2^9 = 512 colorings.
    # We can iterate through all 512 colorings to check for 2-colorability.
    # This is extremely fast.
    for mask in range(1 << n):
        # mask represents the coloring: 0 or 1 for each vertex
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            # Edge is monochromatic if all colors are the same
            if c0 == c1 and c1 == c2:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 60s
    
    # Candidate 1: Fano Plane (7 edges)
    # Vertices 0-6 form Fano plane, 7-8 are isolated.
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify Fano is 3-chromatic (i.e., not 2-colorable)
    if not _is_2_colorable(fano_edges, 9):
        return {'edges': fano_edges, 'colors': fano_colors}
    
    # If Fano is 2-colorable (which it shouldn't be), search for better.
    # But we know it's 3-chromatic. This is a fallback just in case of logic error.
    
    # Candidate 2: Search for 6-edge hypergraph (if it exists and is 3-chromatic)
    # It is a known result that the minimum number of edges for a 3-chromatic
    # 3-uniform hypergraph is 7 (the Fano plane).
    # Therefore, no 6-edge solution exists.
    
    # To be absolutely sure and distinct, we can perform a quick sanity check
    # or just return the Fano plane which is proven optimal.
    
    return {'edges': fano_edges, 'colors': fano_colors}

# EVOLVE-BLOCK-END
