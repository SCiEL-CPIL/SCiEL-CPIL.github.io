"""Optimized Lazebnik-Ustimenko-Woldar construction for 3-regular girth-16 graph."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 130

    q = k  # 3
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    qd = q ** d
    total_vertices = 2 * qd

    # Precompute rule table
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute encoding table: index -> list of d coordinates
    coords = []
    for idx in range(qd):
        x = [0] * d
        tmp = idx
        for i in range(d - 1, -1, -1):
            x[i] = tmp % q
            tmp //= q
        coords.append(tuple(x))

    edges = []
    # For each point, compute its q neighbor lines
    for idx in range(qd):
        x = coords[idx]
        point_vertex = idx  # side 0, index idx

        for y1 in range(q):
            y = [0] * d
            y[0] = y1
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c - 1] = (x[c - 1] + y[a - 1] * x[b - 1]) % q

            # Encode line vertex
            line_idx = 0
            for i in range(d):
                line_idx = line_idx * q + y[i]
            line_vertex = qd + line_idx

            if point_vertex < line_vertex:
                edges.append((point_vertex, line_vertex))
            else:
                edges.append((line_vertex, point_vertex))

    return total_vertices, edges

# EVOLVE-BLOCK-END
