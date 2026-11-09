import time

def construct():
    start_time = time.time()
    n = 16
    left = list(range(0, 8))
    right = list(range(8, 16))

    # Build K_{8,8} as the maximal triangle-free graph on 16 vertices.
    # Mantel's Theorem guarantees no triangle-free graph on 16 vertices has more than 64 edges.
    # K_{8,8} achieves this bound and has diameter 2.
    # Every edge is critical: removing any edge (u, v) with u in left, v in right
    # forces the shortest path between u and v to be length 3 (via any w in left and x in right),
    # raising the diameter to 3.
    edges = []
    for u in left:
        for v in right:
            edges.append([int(u), int(v)])

    # Safety guard: return immediately since construction is O(64)
    if time.time() - start_time > 85:
        return edges

    return edges
