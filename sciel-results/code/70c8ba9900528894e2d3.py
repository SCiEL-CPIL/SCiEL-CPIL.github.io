def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on 16 vertices.
    
    Reasoning:
    1. The task requires a graph with diameter exactly 2.
    2. Every edge must be essential: removing any edge must increase the diameter to > 2 or disconnect the graph.
    3. If a graph is triangle-free, then for any edge (u, v), u and v do not share a common neighbor. 
       Therefore, removing (u, v) increases the distance between u and v to at least 3 (or infinity if disconnected).
       This guarantees that the diameter of the remaining graph is at least 3, satisfying the essentiality condition.
    4. To maximize the number of edges, we should use the densest triangle-free graph on 16 vertices.
    5. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices is floor(n^2 / 4).
       For n = 16, this is 16^2 / 4 = 64 edges.
    6. The unique graph achieving this is the complete bipartite graph K_{8,8}.
    7. K_{8,8} has diameter 2 (vertices in the same partition are distance 2, vertices in different partitions are distance 1).
    8. Thus, K_{8,8} is the optimal solution with 64 edges.
    """
    edges = []
    # Partition vertices into two sets of 8
    # Set A: 0..7, Set B: 8..15
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
