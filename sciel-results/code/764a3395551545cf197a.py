"""LUW graph CD(d,q) - vectorised construction.

The Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k is a k-regular bipartite
graph of girth at least g.  We build the full graph (2*q^d vertices) using
NumPy vectorised arithmetic, which is fast enough to complete within the deadline.
"""

import time
import numpy as np

_T0 = time.time()
_DEADLINE = 120.0


def _rule(d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def construct(k, g):
    t0 = time.time()
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    rule = _rule(d)
    N = q ** d
    n = 2 * N

    # Point coords: all_pts[i] = (x1, ..., xd), index i = x1*q^(d-1) + ... + xd
    all_pts = np.array(np.meshgrid(*[np.arange(q, dtype=np.int64)] * d)).T.reshape(-1, d)

    # Line coords for each (point, y1): y[i, j, c] for c=1..d
    y = np.zeros((N, q, d + 1), dtype=np.int64)
    y[:, :, 1] = np.arange(q, dtype=np.int64)
    for c in range(2, d + 1):
        a = rule[c][0]
        b = rule[c][1]
        xb = all_pts[:, b - 1].reshape(N, 1)
        ya = y[:, :, a]
        y[:, :, c] = (xb + ya * xb) % q

    # Line index
    w = np.array([q ** (d - c) for c in range(1, d + 1)], dtype=np.int64)
    line_idx = np.sum(y[:, :, 1:d + 1] * w.reshape(1, 1, d), axis=2)

    # Adjacency
    E = np.zeros((n, k), dtype=np.int64)
    E[:N, :] = line_idx
    rows = (line_idx + N).reshape(-1)
    cols = np.repeat(np.arange(q, dtype=np.int64), N)
    vals = np.tile(np.arange(N, dtype=np.int64), q)
    E[rows, cols] = vals

    # Convert to edge list
    edges = []
    for i in range(N):
        for j in range(k):
            v = int(E[i, j])
            if i < v:
                edges.append((i, v))
    return n, edges
