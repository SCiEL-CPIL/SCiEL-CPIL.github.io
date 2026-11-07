import time

def construct():
    """
    Constructs a graph on vertices 0..11 consisting of two disjoint K_3,3 graphs.
    This ensures minimum degree 3, triangle-free (bipartite), and contains cycles.
    The distinct cycle lengths in K_3,3 are 4 and 6.
    Thus, the number of distinct cycle lengths is 2.
    """
    start_time = time.time()
    deadline = start_time + 55.0

    edges = []

    # First K_{3,3}: partitions {0,1,2} and {3,4,5}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])

    # Second K_{3,3}: partitions {6,7,8} and {9,10,11}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])

    edges.sort()

    if time.time() > deadline:
        return edges

    result = []
    for e in edges:
        u = e[0]
        v = e[1]
        if isinstance(u, bool) or isinstance(v, bool):
            u = int(u)
            v = int(v)
        else:
            u = int(u)
            v = int(v)
        result.append([u, v])

    return result
