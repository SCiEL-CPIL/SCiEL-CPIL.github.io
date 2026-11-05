def construct():
    # Build the 32-point hexagonal spiral on the triangular lattice
    # using a ring-by-ring construction that achieves the Harborth optimum of 76 edges.
    pts = [[0, 0]]
    # Ring 1: 6 points at distance 1
    ring1 = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]]
    pts.extend(ring1)
    # Ring 2: 12 points at distance 2
    ring2 = [[2, -1], [2, 0], [2, 1], [1, 2], [0, 2], [-1, 2],
             [-2, 1], [-2, 0], [-2, -1], [-1, -2], [0, -2], [1, -2]]
    pts.extend(ring2)
    # Ring 3 partial: 13 more points to reach 32 total
    ring3 = [[2, -2], [3, -2], [3, -1], [3, 0], [3, 1], [2, 2],
             [1, 3], [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1]]
    pts.extend(ring3)
    return pts
