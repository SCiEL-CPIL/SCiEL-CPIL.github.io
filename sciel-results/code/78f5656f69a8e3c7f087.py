import time

def construct():
    start = time.time()
    n = 16
    half = n // 2
    # Construct K_{8,8}
    # Vertices 0..7 in left partition, 8..15 in right partition
    # K_{8,8} has diameter 2; deleting any edge (u,v) makes dist(u,v)=3 > 2.
    # By Mantel's theorem, max triangle-free edges on 16 vertices is 64.
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    # Safety margin: return before 90s wall-clock limit
    if time.time() - start > 85:
        return edges
    return edges
