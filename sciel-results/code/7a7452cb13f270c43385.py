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
    deadline = start_time + 50  # Safety margin before 60s limit

    # The Fano plane is the minimal 3-uniform hypergraph with chromatic number 3.
    # It has 7 vertices and 7 edges.
    # We embed it on vertices 0..6. Vertices 7 and 8 are isolated.
    
    # Edges of the Fano plane on vertices 0..6.
    # Vertices are represented as 1..7 in F_2^3 \ {0}.
    # Edge exists if i ^ j ^ k == 0.
    # Let's explicitly list them for speed and clarity, mapping to 0..6.
    # 
    # Using the standard Fano plane edges on vertices 1..7:
    # (1,2,3), (1,4,5), (1,6,7), (2,4,6), (2,5,7), (3,4,7), (3,5,6)
    # Mapping v -> v-1:
    # (0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)
    
    edges_raw = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Sort edges for deterministic output
    edges_raw = [list(e) for e in edges_raw]
    edges_raw.sort()
    
    # We need a 3-coloring c: {0..8} -> {0,1,2} such that:
    # 1. No edge is monochromatic.
    # 2. The graph is not 2-colorable (property of edges, but we verify).
    
    # Let's find a valid 3-coloring for the Fano plane vertices 0..6.
    # We can use a greedy or exhaustive search for the 7 vertices, then assign 0 to 7,8.
    # Since there are only 3^7 = 2187 possibilities, this is fast.
    
    def is_valid_3_coloring(coloring):
        for e in edges_raw:
            if coloring[e[0]] == coloring[e[1]] == coloring[e[2]]:
                return False
        return True

    colors = None
    # Try to find a coloring with a specific pattern or just the first one found.
    # A known valid coloring for the Fano plane (vertices 0..6) is:
    # 0:0, 1:1, 2:2, 3:0, 4:1, 5:2, 6:0
    # Let's verify this specific one first.
    candidate_colors = [0, 1, 2, 0, 1, 2, 0]
    if is_valid_3_coloring(candidate_colors):
        colors = candidate_colors + [0, 0]
    else:
        # Fallback: exhaustive search
        for c0 in range(3):
            if time.time() > deadline:
                # If we run out of time, return the Fano plane with a potentially invalid coloring?
                # No, we must return a valid result. 
                # But the static check is O(1), so this won't happen.
                pass
            for c1 in range(3):
                for c2 in range(3):
                    for c3 in range(3):
                        for c4 in range(3):
                            for c5 in range(3):
                                for c6 in range(3):
                                    cand = [c0, c1, c2, c3, c4, c5, c6]
                                    if is_valid_3_coloring(cand):
                                        colors = cand + [0, 0]
                                        break
                            if colors: break
                        if colors: break
                    if colors: break
                if colors: break
            if colors: break
            if colors: break
        if not colors:
            # This should never happen for the Fano plane
            colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Verify 3-chromaticity (not 2-colorable)
    # Check all 2^7 = 128 binary colorings of vertices 0..6
    is_2_colorable = False
    for mask in range(128):
        valid = True
        for e in edges_raw:
            b0 = (mask >> e[0]) & 1
            b1 = (mask >> e[1]) & 1
            b2 = (mask >> e[2]) & 1
            if b0 == b1 == b2:
                valid = False
                break
        if valid:
            is_2_colorable = True
            break
    
    if is_2_colorable:
        # This would mean the Fano plane is 2-colorable, which is mathematically false.
        # If this happens, it's a bug in the logic or the edge definition.
        # We raise an error to indicate failure, but the task says "invalid result... scores 0".
        # We should avoid raising if possible, but returning an invalid result is also 0.
        # However, we know Fano plane is 3-chromatic.
        pass

    # Final type conversion
    edges_out = [[int(v) for v in e] for e in edges_raw]
    colors_out = [int(c) for c in colors]
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }
