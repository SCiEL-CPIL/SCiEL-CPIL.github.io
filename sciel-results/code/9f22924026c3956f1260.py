def construct():
    # Use (i + j) mod 13.
    # 13 is odd, so no two disjoint pairs of edges in any K4 share the same color.
    # Adjacent edges never share a color.
    # Thus, every K4 has at most 1 color collision, implying at least 5 distinct colors.
    # The number of colors used is 13.
    return [(i + j) % 13 for i in range(16) for j in range(i + 1, 16)]
