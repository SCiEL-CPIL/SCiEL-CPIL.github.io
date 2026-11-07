"""
Constructs a 3-uniform hypergraph on 9 vertices (0-8) that is not 2-colorable
with the minimum number of edges.

The Fano plane is the unique (up to isomorphism) minimal 3-uniform non-2-colorable
hypergraph, having 7 edges. It is a well-known result that any 3-uniform hypergraph
with fewer than 7 edges is 2-colorable, so 7 is the minimum edge count for a
3-chromatic triple system.

The program returns the Fano plane edges on vertices 0-6 with vertices 7 and 8
isolated, along with a valid 3-coloring where every edge is non-monochromatic.
"""

import time


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Fano plane edges on vertices 0-6.
    # This is the unique minimal 3-uniform non-2-colorable hypergraph (7 edges).
    fano_edges = (
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    )

    # Verify: no two edges are the same (distinct triples)
    edge_set = set(fano_edges)
    assert len(edge_set) == 7, "Edges must be distinct"

    # A valid 3-coloring for the Fano plane using colors 0, 1, 2.
    # Assignment: v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1
    # Check each edge is non-monochromatic:
    # (0,1,2): 0,1,2 ✓
    # (0,3,4): 0,1,2 ✓
    # (0,5,6): 0,0,1 ✓
    # (1,3,5): 1,1,0 ✓
    # (1,4,6): 1,2,1 ✓
    # (2,3,6): 2,1,1 ✓
    # (2,4,5): 2,2,0 ✓
    fano_colors = (0, 1, 2, 1, 2, 0, 1, 0, 0)

    # Verify coloring: every edge must be non-monochromatic
    for edge in fano_edges:
        c0, c1, c2 = fano_colors[edge[0]], fano_colors[edge[1]], fano_colors[edge[2]]
        assert not (c0 == c1 == c2), f"Edge {edge} is monochromatic"

    # Verify: the coloring uses at most 3 colors (labels from 0..2)
    assert all(0 <= c <= 2 for c in fano_colors)

    # Convert to lists of Python integers to satisfy type constraints
    edges = [tuple(int(v) for v in edge) for edge in fano_edges]
    colors = [int(c) for c in fano_colors]

    # Explicit deadline guard
    if time.time() > deadline:
        pass

    return {
        'edges': edges,
        'colors': colors
    }
