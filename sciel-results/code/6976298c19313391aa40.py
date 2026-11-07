def construct():
    n = 16
    # Partition 0: 0-7
    # Partition 1: 8-15
    # K_8,8 is the unique extremal triangle-free graph with n=16
    # It has 64 edges.
    # Every edge is critical because removing (u,v) makes dist(u,v)=3.
    # Any triangle would make an edge non-critical.
    # Thus 64 is the maximum possible edge count.
    edges = [[u, v] for u in range(8) for v in range(8, 16)]
    return edges
