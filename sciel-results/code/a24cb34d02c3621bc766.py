"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import Counter

def _score(points):
    """Compute the max multiplicity of a positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        det = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if det > 0:
            areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def construct():
    # Construct 15 points on y=0 and 15 points on y=1
    # A_i = (i, 0) for i in 0..14
    # B_j = (j, 1) for j in 0..14
    
    points = []
    for i in range(15):
        points.append((i, 0))
    for j in range(15):
        points.append((j, 1))
        
    # Verify constraints
    if len(points) != 30:
        return [(x, y) for x in range(6) for y in range(5)]
    if len(set(points)) != 30:
        return [(x, y) for x in range(6) for y in range(5)]
    if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in points):
        return [(x, y) for x in range(6) for y in range(5)]
        
    return points
# EVOLVE-BLOCK-END
