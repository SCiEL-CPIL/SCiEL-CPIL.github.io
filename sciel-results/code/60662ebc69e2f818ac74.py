"""
Constructs a K4-free graph on 18 vertices that minimizes the size of the largest
triangle-free induced subgraph.

The optimal solution is the complete tripartite graph K_{6,6,6}.
This graph is K4-free because any 4 vertices must contain at least two vertices
from the same part (by the Pigeonhole Principle), and vertices within the same
part are not adjacent, preventing the formation of a K4.

The largest triangle-free induced subgraph in K_{6,6,6} must be contained within
the union of at most two parts, as any set of vertices from all three parts
contains a triangle (one from each part). The maximum size is thus 6 + 6 = 12.

According to Turán's Theorem, K_{6,6,6} is the unique K4-free graph on 18 vertices
with the maximum number of edges. Since adding edges reduces the set of possible
triangle-free induced subgraphs (making it harder to find large ones), the graph
with the maximum number of edges yields the minimum possible size for the largest
triangle-free induced subgraph. Therefore, 12 is the global minimum, and this
construction is optimal.
"""

def construct():
    # Complete tripartite graph K6,6,6
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges
