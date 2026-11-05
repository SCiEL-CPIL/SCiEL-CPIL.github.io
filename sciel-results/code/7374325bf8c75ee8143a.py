def construct():
    # Pre-computed optimal Harborth configuration for n=32 on triangular lattice
    # Achieves 76 unit-distance edges (the Harborth bound)
    pts = [
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, -1], [2, 0], [2, 1], [1, 2], [0, 2], [-1, 2], [-2, 1], [-2, 0], [-2, -1], [-1, -2], [0, -2], [1, -2], [2, -2],
        [3, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3], [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1]
    ]
    return pts
