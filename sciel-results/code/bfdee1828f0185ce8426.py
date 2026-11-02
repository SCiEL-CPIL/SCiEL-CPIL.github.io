"""Construct a k-regular graph of girth at least g using the full
Lazebnik-Ustimenko-Woldar (LUW) graph CD(d, q).

The seed returned only the connected component of the zero vertex, which is
a small fraction of the graph. The complete LUW graph has q**(d+2) vertices,
is q-regular, simple, and has girth at least d+5 (odd d). Returning the full
graph (all vertices) yields a much larger valid instance.
"""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin under the 150s limit
    q = k
    # choose odd d >= 3 with d + 5 >= g
    d = max(3, g - 5)
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3

    # field arithmetic over GF(q) (q prime, or q = 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # LUW rule: for c in 2..d, (a, b) with a < c < b
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # enumerate all vertices: side 0 (points) and side 1 (lines),
    # each coordinate 0..d in range q -> q**(d+2) total vertices
    # index = side * q**(d+1) + coord0*q**d + coord1*q**(d-1) + ... + coordd
    qpow = [1] * (d + 2)
    for i in range(1, d + 2):
        qpow[i] = qpow[i - 1] * q
    half = qpow[d + 1]          # q**(d+1): number of vertices per side
    total = 2 * half            # q**(d+2)

    # decode a flat index into (side, coord tuple of length d+1)
    def decode(idx):
        side = 1 if idx >= half else 0
        r = idx - (half if side else 0)
        coords = [0] * (d + 1)
        for c in range(d, -1, -1):
            coords[c] = r % q
            r //= q
        return side, coords

    # for a vertex, compute its q neighbours (on the opposite side)
    def neighbours(idx):
        side, x = decode(idx)
        # x is a list; x[0] is the "first" coordinate we iterate over
        out = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] = p[c] + l[a]*p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: p[c] = l[c] - l[a]*p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            # encode the neighbour on the opposite side
            nidx = (1 - side) * half
            for c in range(d, -1, -1):
                nidx = nidx * q + y[c]
            out.append(nidx)
        return out

    # build the edge list: for each point (side 0), add edges to its neighbours
    # (each edge appears exactly once because we only emit from side 0)
    edges = []
    for idx in range(half):  # side 0 vertices only
        for nb in neighbours(idx):
            edges.append((idx, nb))

    return total, edges
# EVOLVE-BLOCK-END
