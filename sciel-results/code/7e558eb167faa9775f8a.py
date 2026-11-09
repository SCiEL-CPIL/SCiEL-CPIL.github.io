"""
Constructs a graph on vertices 0..11 with minimum degree 3, girth at least 4,
and exactly 2 distinct simple cycle lengths (4 and 6).

Uses two disjoint K_{3,3} components (bipartite, min degree 3, triangle-free).
The cycle lengths present are exactly {4, 6}, giving raw value 2.
Score = seed/value = 5/2 = 2.5.
"""

import time

_START_TIME = None


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Build two disjoint K_{3,3} components
    # Partition A: vertices 0-5, Partition B: vertices 6-11
    # Component 1: {0,1,2} x {6,7,8}
    # Component 2: {3,4,5} x {9,10,11}

    edges = []

    # Component 1
    for u in (0, 1, 2):
        for v in (6, 7, 8):
            edges.append([u, v])

    # Component 2
    for u in (3, 4, 5):
        for v in (9, 10, 11):
            edges.append([u, v])

    # Deduplicate and validate (no loops, no duplicates)
    seen = set()
    final_edges = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        final_edges.append([u, v])

    final_edges.sort(key=lambda x: (x[0], x[1]))

    return final_edges
