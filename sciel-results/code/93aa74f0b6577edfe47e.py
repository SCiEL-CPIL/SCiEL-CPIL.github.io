# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a simple undirected graph on vertices 0..15 with diameter exactly 2,
    where every edge is essential (removing any edge increases diameter > 2 or disconnects).
    Maximizes edge count.
    
    Analysis:
    1. The condition that removing any edge increases diameter > 2 or disconnects
       implies the graph must be triangle-free.
       Proof: If a triangle (u,v,w) exists, removing (u,v) leaves path u-w-v of length 2.
       Dist(u,v) remains 2. If graph was diameter 2, it stays diameter 2 (not > 2)
       and remains connected. Thus, edge (u,v) is not "essential" in the required sense.
    2. By Mantel's Theorem, the maximum number of edges in a triangle-free graph
       on n vertices is floor(n^2/4).
    3. For n=16, max edges = 16^2 / 4 = 64.
    4. This maximum is achieved by the complete bipartite graph K_{8,8}.
    5. K_{8,8} has diameter 2.
       - Vertices in same partition have distance 2.
       - Vertices in different partitions have distance 1.
       - Max distance is 2.
    6. K_{8,8} is triangle-free, so all edges are essential.
    
    Therefore, K_{8,8} is the optimal solution with 64 edges.
    """
    _t0 = time.time()
    _deadline = _t0 + 85.0  # Safety margin before 90s kill

    n = 16
    # Partition vertices into two sets of size 8
    # A: 0..7, B: 8..15
    edges = []
    
    # Construct K_{8,8}
    # Edge (u, v) exists if u in A and v in B
    for u in range(8):
        for v in range(8, 16):
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    # Deadline guard
    if time.time() > _deadline:
        return edges
    
    return edges
# EVOLVE-BLOCK-END
