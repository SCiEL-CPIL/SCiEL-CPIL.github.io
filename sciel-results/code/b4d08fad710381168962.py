"""Program for 'Few hyperedges forcing three colors'.

Uses a Min-SAT / iterative deepening search over edge subsets to find the
minimum number of distinct 3-uniform edges on vertices 0..8 such that no
proper 2-coloring exists, while every edge is non-monochromatic under a
supplied 3-coloring.  The result is the optimal construction (7 edges,
the Fano plane embedded on vertices 0..6).
"""

# EVOLVE-BLOCK-START
import itertools


def _all_triples():
    """Return all distinct 3-subsets of {0..8} as tuples, deterministically ordered."""
    return [tuple(sorted(t)) for t in itertools.combinations(range(9), 3)]


def _edge_monochromatic(edge, color_bits):
    """Return True iff all vertices in edge have the same 2-color bit."""
    return (
        (color_bits >> edge[0]) & 1
        == (color_bits >> edge[1]) & 1
        == (color_bits >> edge[2]) & 1
    )


def _has_proper_2coloring(edges):
    """Exhaustive check: does a 2-coloring exist in which no edge is monochromatic?"""
    n = 9
    for bits in range(1 << n):
        ok = True
        for e in edges:
            if (
                (bits >> e[0]) & 1
                == (bits >> e[1]) & 1
                == (bits >> e[2]) & 1
            ):
                ok = False
                break
        if ok:
            return True
    return False


def _find_minimal_3chromatic_edges():
    """
    Deterministic iterative deepening search:
      For k = 1, 2, 3, ...:
        For each subset S of all triples of size k (in lexicographic order):
          If no proper 2-coloring exists for S, return S.
    This guarantees minimum edge count.
    """
    triples = _all_triples()
    k = 1
    while True:
        # For each combination of k triples, check 2-colorability
        for combo in itertools.combinations(triples, k):
            if not _has_proper_2coloring(combo):
                return list(combo)
        k += 1


def _color_for_3chromatic(edges):
    """
    Find a 3-coloring (labels 0..2) on vertices 0..8 such that every edge
    in edges is non-monochromatic.  Uses backtracking search.
    """
    n = 9

    def valid_partial(coloring, idx):
        # Check all edges whose max vertex index is <= idx
        for e in edges:
            max_idx = max(e)
            if max_idx <= idx:
                c = {coloring[v] for v in e}
                if len(c) == 1:
                    return False
        return True

    def backtrack(idx, coloring):
        if idx == n:
            return coloring
        for c in range(3):
            coloring[idx] = c
            if valid_partial(coloring, idx):
                result = backtrack(idx + 1, coloring)
                if result is not None:
                    return result
        return None

    return backtrack(0, [None] * n)


def construct():
    # Find minimal set of edges forcing 3-chromaticity via Min-SAT / deepening
    edges = _find_minimal_3chromatic_edges()
    # Convert to list of lists for return
    edges = [list(e) for e in edges]

    # Find a valid 3-coloring
    colors = _color_for_3chromatic(edges)

    # --- Runtime verification ---
    assert len(set(map(tuple, edges))) == len(edges), "edges not distinct"
    for e in edges:
        assert len(set(e)) == 3, "edge has repeated vertex"
        assert all(isinstance(v, int) and 0 <= v <= 8 for v in e), "vertex out of range"
    assert len(colors) == 9, "colors must have 9 entries"
    assert all(isinstance(c, int) and c in (0, 1, 2) for c in colors), "color labels invalid"
    for e in edges:
        c = {colors[v] for v in e}
        assert len(c) >= 2, "edge is monochromatic under 3-coloring"
    assert not _has_proper_2coloring(edges), "a proper 2-coloring exists"

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
