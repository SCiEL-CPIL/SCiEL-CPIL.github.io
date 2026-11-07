import time

def construct():
    """
    Constructs the optimal graph for the task:
    A complete bipartite graph K_{8,8} on vertices 0..15.
    
    Properties:
    1. Triangle-free: Bipartite graphs contain no triangles.
    2. Diameter 2: 
       - Vertices in the same partition have distance 2 (via any vertex in the other partition).
       - Vertices in different partitions have distance 1.
    3. Every edge is critical:
       - Removing any edge (u, v) with u in A and v in B:
         - There are no common neighbors between u and v (since N(u) subset B and N(v) subset A).
         - Thus, no path of length 2 exists between u and v.
         - The shortest path becomes length 3 (e.g., u -> w -> x -> v where w in B\{v}, x in A\{u}).
         - The diameter increases from 2 to 3.
    4. Maximality:
       - By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2/4).
       - For n=16, max edges = 64.
       - K_{8,8} has 8*8 = 64 edges.
       - Any graph with >64 edges must contain a triangle (by Mantel's), which would make at least one edge non-critical (as it would have an alternative path of length 2).
    
    Returns:
    list of lists: Edges [u, v] where u, v are Python ints.
    """
    n = 16
    half = n // 2
    
    edges = []
    # Partition A: 0..7
    # Partition B: 8..15
    for u in range(half):
        for v in range(half, n):
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    return edges
