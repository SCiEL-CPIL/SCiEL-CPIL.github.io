def construct():
    # Maximum edge-critical diameter-2 graph on 16 vertices.
    # By Mantel's theorem, any triangle-free graph on 16 vertices has at most 64 edges.
    # K_{8,8} achieves this bound and every edge is critical for diameter 2.
    # Returning the edge list as Python ints in lists.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
    return edges
