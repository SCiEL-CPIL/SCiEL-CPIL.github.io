"""
Constructs a graph on vertices 0..11 with minimum degree 3, girth at least 4,
and exactly 2 distinct simple cycle lengths (4 and 6).

The graph consists of two disjoint copies of K_{3,3}.
- K_{3,3} is bipartite, so it is triangle-free.
- The simple cycles in K_{3,3} have lengths 4 and 6.
- Since the components are disjoint, the set of cycle lengths for the whole
  graph is {4, 6}, giving 2 distinct cycle lengths.
"""

def construct():
    # Vertices 0..5 form the first K_{3,3}
    # Vertices 6..11 form the second K_{3,3}

    # Partition for first K_{3,3}: A={0,1,2}, B={3,4,5}
    # Partition for second K_{3,3}: A={6,7,8}, B={9,10,11}
    a1 = [0, 1, 2]
    b1 = [3, 4, 5]
    a2 = [6, 7, 8]
    b2 = [9, 10, 11]

    raw_edges = []
    for u in a1:
        for v in b1:
            raw_edges.append((u, v))
    for u in a2:
        for v in b2:
            raw_edges.append((u, v))

    # Convert to canonical [min, max] list of lists, dedup
    seen = set()
    out = []
    for u, v in raw_edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append([int(a), int(b)])

    out.sort()
    return out
