import time


def construct():
    """
    Constructs a triangle-free graph on vertices 0..11 with minimum degree 3
    that minimizes the number of distinct simple cycle lengths.

    The optimal value is 2, achieved by two disjoint K_{3,3} components.
    K_{3,3} has only cycle lengths 4 and 6.
    Thus, the set of cycle lengths is {4, 6}, size 2.

    This is provably optimal: any 3-regular bipartite graph on 12 vertices
    must contain cycles of at least two distinct lengths.
    """
    deadline = time.time() + 58  # Safety margin before 60s limit

    edges = []

    # First K_{3,3}: vertices 0,1,2 in U, 3,4,5 in V
    U1 = [0, 1, 2]
    V1 = [3, 4, 5]
    for u in U1:
        for v in V1:
            edges.append([u, v])

    # Second K_{3,3}: vertices 6,7,8 in U, 9,10,11 in V
    U2 = [6, 7, 8]
    V2 = [9, 10, 11]
    for u in U2:
        for v in V2:
            edges.append([u, v])

    # Ensure all elements are Python ints in lists
    edges = [[int(u), int(v)] for u, v in edges]

    return edges
