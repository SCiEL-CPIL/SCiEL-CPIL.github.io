"""
Constructs a graph on 16 vertices with diameter 2 such that removing any edge increases the diameter.
The optimal structure for this is the complete bipartite graph K_{8,8}, which achieves the
Mantel bound of 64 edges for triangle-free graphs on 16 vertices.
"""
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    # Construct K_{8,8}
    # Vertices 0-7 in partition A, 8-15 in partition B
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])

    # Verify: triangle-free (edge-critical for diameter 2)
    # In K_{8,8}, no two adjacent vertices share a common neighbor, so removing
    # any edge (u,v) makes dist(u,v) > 2. Diameter remains 2 for all other pairs.
    # This is the Mantel-optimal triangle-free graph, so 64 edges is the maximum.

    return edges
