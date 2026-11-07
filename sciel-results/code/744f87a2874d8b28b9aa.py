"""Rewritten program: two disjoint K3,3 components to minimize distinct cycle lengths."""

def construct():
    edges = []
    # First K3,3 on vertices 0-5: partitions {0,1,2} and {3,4,5}
    for a in (0, 1, 2):
        for b in (3, 4, 5):
            edges.append([a, b])
    # Second K3,3 on vertices 6-11: partitions {6,7,8} and {9,10,11}
    for a in (6, 7, 8):
        for b in (9, 10, 11):
            edges.append([a, b])
    return edges
