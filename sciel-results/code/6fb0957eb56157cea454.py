# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    """
    Construct K_{8,8} complete bipartite graph on vertices 0..15.
    
    Properties:
    - Simple undirected graph with 16 vertices and 64 edges.
    - Diameter is exactly 2 (any two vertices in different partitions are distance 1, 
      any two in the same partition are distance 2 via any vertex in the other partition).
    - Every edge is essential: deleting any edge (u,v) makes the distance between u and v 
      become 3 (u - w - x - v where w in u's partition, x in v's partition), which is > 2.
      Thus the diameter increases above 2 for every single edge deletion.
    - By Mantel's theorem, K_{8,8} is the unique triangle-free graph with the maximum 
      number of edges (64) on 16 vertices. Since the graph must be triangle-free to 
      maintain edge-criticality for diameter 2 (triangles would create redundant 2-paths), 
      this achieves the theoretical maximum edge count.
    """
    start = time.time()
    n = P['n']
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    
    # Safety margin: return before 90s wall-clock limit
    # This construction is O(n^2) and completes in microseconds, so this guard is purely defensive.
    if time.time() - start > 85:
        return edges
    return edges
# EVOLVE-BLOCK-END
