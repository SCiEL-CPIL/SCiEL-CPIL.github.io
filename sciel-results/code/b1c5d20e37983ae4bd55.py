"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    import time
    _start = time.time()
    _DEADLINE = 80.0  # safety margin before 90s kill

    # The Fano plane is the proven unique minimum 3-uniform hypergraph
    # with chromatic number 3. It has exactly 7 edges.
    # No 3-uniform hypergraph with fewer than 7 edges is 3-chromatic,
    # regardless of the number of vertices (vertices 7,8 are isolated).
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring: no edge is monochromatic
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify no edge is monochromatic (fast sanity check)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        assert not (c0 == c1 == c2), f"Monochromatic edge {e}"
    
    # Ensure all types are strictly Python native
    edges_out = [tuple(int(v) for v in e) for e in edges]
    colors_out = tuple(int(c) for c in colors)
    
    # Deadline guard: we're done, but check before returning
    if time.time() - _start > _DEADLINE:
        # Should never happen for this constant solution
        pass
    
    return {
        'edges': edges_out,
        'colors': colors_out
    }

# EVOLVE-BLOCK-END
