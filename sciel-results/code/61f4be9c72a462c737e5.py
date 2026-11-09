"""Construct a minimal 3-uniform 3-chromatic hypergraph on 9 vertices.

Returns the Fano plane (7 edges) on vertices 0-6, which is the theoretical
minimum for a 3-chromatic 3-uniform hypergraph. Vertices 7 and 8 are isolated.
"""

def construct():
    # Fano plane edges on vertices 0..6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]

    # Valid 3-coloring: no edge is monochromatic.
    # Vertices 7, 8 are isolated; their colors are arbitrary.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # --- Verification: provided coloring is valid ---
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            raise ValueError(f"Edge {e} is monochromatic under provided colors")

    # --- Verification: hypergraph is NOT 2-colorable ---
    # Enumerate all 2^9 = 512 two-colorings.
    # A 2-coloring "works" if every edge is non-monochromatic.
    # We must confirm NO such coloring exists.
    n = 9
    for mask in range(1 << n):
        # Extract color for each vertex from the bitmask
        c = [int((mask >> i) & 1) for i in range(n)]
        ok = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                ok = False
                break
        if ok:
            raise ValueError("Hypergraph is 2-colorable — logic error")

    # --- Return with exact types ---
    # Ensure edges is a list of tuples of Python ints
    result_edges = [tuple(int(v) for v in e) for e in edges]
    # Ensure colors is a list of Python ints
    result_colors = [int(c) for c in colors]

    # Final type assertions
    assert isinstance(result_edges, list)
    assert len(result_edges) >= 1
    for e in result_edges:
        assert isinstance(e, tuple)
        assert len(e) == 3
        for v in e:
            assert isinstance(v, int) and not isinstance(v, bool)
            assert 0 <= v <= 8
    assert isinstance(result_colors, list)
    assert len(result_colors) == 9
    for c in result_colors:
        assert isinstance(c, int) and not isinstance(c, bool)
        assert c in (0, 1, 2)

    return {
        'edges': result_edges,
        'colors': result_colors
    }
