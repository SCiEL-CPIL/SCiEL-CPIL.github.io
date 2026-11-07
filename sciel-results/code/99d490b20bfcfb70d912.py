"""Deterministic optimal 3-chromatic triple system on 9 vertices (Fano plane core)."""

import time

def construct():
    """Return a dict with 'edges' and 'colors' for a minimal 3-chromatic triple system.

    The Fano plane on 7 vertices is the unique 3-uniform hypergraph on 7 vertices
    with chromatic number 3. Any 3-uniform hypergraph with fewer than 7 edges is
    2-colorable, so 7 is the global minimum. Vertices 7 and 8 are isolated.
    """
    deadline = time.time() + 55.0  # safety margin before 60s limit

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

    # Valid 3-coloring: v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1, v7=0, v8=0
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Ensure all types are Python int (not numpy, not bool)
    edges = [[int(x) for x in e] for e in edges]
    colors = [int(c) for c in colors]

    if time.time() > deadline:
        pass  # shouldn't happen, but guard

    return {'edges': edges, 'colors': colors}
