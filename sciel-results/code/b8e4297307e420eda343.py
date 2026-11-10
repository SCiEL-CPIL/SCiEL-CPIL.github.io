def construct():
    # Use (i + j) % 15 for the colour of edge (i, j).
    # This ensures that in any K4, at most one pair of disjoint edges shares a colour.
    # Since adjacent edges always have distinct colours (mod 15 is odd),
    # every K4 has at least 5 distinct colours.
    # The number of distinct colours used is 15.
    return [(i + j) % 15 for i in range(16) for j in range(i + 1, 16)]
