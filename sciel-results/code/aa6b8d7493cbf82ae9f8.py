"""Rewritten program for the few-cycle-lengths task.

The optimal construction is two disjoint K_{3,3} components.
K_{3,3} is bipartite, 3-regular, and girth 4. Its only simple cycles
have lengths 4 and 6. Using two disjoint copies on vertices 0..11 keeps
the distinct cycle-length set exactly {4, 6}, giving raw value 2.
"""

# EVOLVE-BLOCK-START


def construct():
    """Return the edge list for two disjoint K_{3,3} graphs on vertices 0..11.

    Component 1: bipartition L1={0,1,2}, R1={3,4,5}
    Component 2: bipartition L2={6,7,8}, R2={9,10,11}

    Every vertex has degree exactly 3. The graph is triangle-free (bipartite)
    and contains 4- and 6-cycles. No other cycle lengths appear.
    """
    edges = []

    # First K3,3 component
    L1 = [0, 1, 2]
    R1 = [3, 4, 5]
    for u in L1:
        for v in R1:
            edges.append([u, v])

    # Second K3,3 component
    L2 = [6, 7, 8]
    R2 = [9, 10, 11]
    for u in L2:
        for v in R2:
            edges.append([u, v])

    # Ensure all elements are Python int and edges are lists of two ints
    result = []
    for e in edges:
        a = int(e[0])
        b = int(e[1])
        result.append([a, b])

    return result


# EVOLVE-BLOCK-END
