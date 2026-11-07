"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 5s margin before 60s limit

    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges and 7 vertices.
    # We map the Fano plane to vertices 0..6.
    # Vertices 7 and 8 are isolated, so they do not affect the chromatic number
    # (they can be colored arbitrarily) and do not add to the edge count.
    
    # Fano plane edges (triples)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6.
    # We verify this coloring manually:
    # v0: 0, v1: 1, v2: 2, v3: 1, v4: 2, v5: 0, v6: 1
    # Edge [0,1,2]: 0,1,2 -> OK
    # Edge [0,3,4]: 0,1,2 -> OK
    # Edge [0,5,6]: 0,0,1 -> OK
    # Edge [1,3,5]: 1,1,0 -> OK
    # Edge [1,4,6]: 1,2,1 -> OK
    # Edge [2,3,6]: 2,1,1 -> OK
    # Edge [2,4,5]: 2,2,0 -> OK
    
    # Vertices 7 and 8 are isolated, assign color 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify non-monochromatic edges for the provided coloring
    for edge in edges:
        if colors[edge[0]] == colors[edge[1]] == colors[edge[2]]:
            # This should not happen. If it does, the solution is invalid.
            # We raise an error to avoid returning an invalid solution, 
            # though this should never trigger for the static Fano plane data.
            raise ValueError("Invalid coloring: monochromatic edge found")

    # Verify 3-chromaticity (i.e., not 2-colorable)
    # This is a known mathematical fact for the Fano plane.
    # We perform a quick check if time permits to be robust against any potential error.
    # The check involves iterating over 2^9 = 512 colorings.
    
    if time.time() < deadline:
        is_2_colorable = False
        for c in itertools.product([0, 1], repeat=9):
            valid = True
            for e in edges:
                if c[e[0]] == c[e[1]] == c[e[2]]:
                    valid = False
                    break
            if valid:
                is_2_colorable = True
                break
        
        if is_2_colorable:
            # This would indicate the edge set is incorrect for a 3-chromatic graph.
            # Since we are using the standard Fano plane, this should never happen.
            # If it does, the result is invalid.
            raise ValueError("Hypergraph is 2-colorable, chromatic number is not 3")

    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
