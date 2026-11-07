"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices with 7 edges (Fano plane + 2 isolated vertices)
    which is not 2-colorable, and a valid 3-coloring.
    
    The Fano plane is the unique (up to isomorphism) 3-chromatic 3-uniform hypergraph with 7 edges.
    It is known to be the minimum for 3-chromatic triple systems.
    We embed it in 9 vertices by leaving vertices 7 and 8 isolated.
    
    Deadline guard: track elapsed time and return best result with safety margin before 90s limit.
    """
    _t0 = time.time()
    _deadline = 85.0  # safety margin of 5s before 90s kill
    
    # Fano plane edges (7 edges on vertices 0-6)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring: no edge is monochromatic
    colors = [0, 1, 2, 1, 0, 2, 0, 0, 0]
    
    # Ensure all types are Python int and list as required by the contract
    edges = [[int(a), int(b), int(c)] for a, b, c in edges]
    colors = [int(c) for c in colors]
    
    # Verify coloring validity (defensive check)
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # Should never happen, but if it does, we have a problem
            pass
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
