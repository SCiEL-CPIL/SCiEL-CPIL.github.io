"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55.0

    # The Fano plane (7 edges on 7 vertices) is a 3-chromatic 3-uniform hypergraph.
    # Adding two isolated vertices (7,8) keeps it 3-chromatic and 3-colorable.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]

    # Quick sanity: every edge non-monochromatic under the given coloring
    for e in edges:
        if len({colors[v] for v in e}) < 2:
            # Fallback (should not happen)
            pass

    return {'edges': [list(e) for e in edges], 'colors': [int(c) for c in colors]}
# EVOLVE-BLOCK-END
