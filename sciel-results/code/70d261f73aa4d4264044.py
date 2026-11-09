"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Create a 2x7 grid of points
    # Points are (x, y) where x in {0, 1} and y in {0, 1, 2, 3, 4, 5, 6}
    # We need 13 points, so we remove one corner point (0,0)
    points = []
    for x in [0, 1]:
        for y in range(7):
            if x == 0 and y == 0:
                continue
            points.append([x, y])
    
    return np.array(points)
# EVOLVE-BLOCK-END
