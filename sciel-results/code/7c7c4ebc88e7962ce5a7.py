"""
Constructs a dense graph with diameter 2 where every edge is essential.

The problem requires a graph on 16 vertices with:
1. Diameter exactly 2.
2. Every edge is essential: removing any single edge increases the diameter to > 2 or disconnects the graph.
3. Maximize the number of edges.

Key insight:
- If an edge (u, v) has a common neighbor w (forming triangle u-w-v), then removing (u, v)
  leaves a path u-w-v of length 2. Since the original diameter is 2, all other distances
  remain <= 2 after removal. Thus the new diameter stays 2, violating the "essential" requirement.
- Therefore, the graph must be triangle-free.
- By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n=16 vertices
  is floor(16^2 / 4) = 64, achieved uniquely by K_{8,8}.
- K_{8,8} has diameter 2 and is triangle-free, so every edge is essential.
- Hence K_{8,8} is the unique optimal solution with 64 edges.
"""

def construct():
    """
    Returns the edges of the complete bipartite graph K_{8,8} on 16 vertices.
    This graph has 64 edges, diameter 2, and is triangle-free, making every edge essential.
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
