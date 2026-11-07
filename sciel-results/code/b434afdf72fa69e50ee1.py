"""Construct a graph on 12 vertices with minimum degree >= 3, girth >= 4,
minimizing the number of distinct simple cycle lengths.

Strategy: two disjoint K_{3,3} components.
Each K_{3,3} is bipartite with parts of size 3, so all cycles have even length
2k where 2 <= k <= 3, i.e., cycle lengths are exactly {4, 6}.
The union has the same set of cycle lengths: {4, 6} → 2 distinct lengths.
"""

def construct():
    edges = []
    # Component 1: K_{3,3} on vertices 0,1,2 (part A) and 3,4,5 (part B)
    for i in range(3):
        for j in range(3, 6):
            u, v = i, j
            edges.append([min(u, v), max(u, v)])
    # Component 2: K_{3,3} on vertices 6,7,8 (part A) and 9,10,11 (part B)
    for i in range(6, 9):
        for j in range(9, 12):
            u, v = i, j
            edges.append([min(u, v), max(u, v)])
    return edges
