import random
import numpy as np
from collections import Counter
from itertools import combinations

random.seed(42)

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area (absolute determinant).
    """
    # Strategy: Use 3 parallel lines y=0, y=1, y=2.
    # Place 10 points on each line with x-coordinates 0..9.
    # This configuration maximizes the number of triangles with small doubled areas,
    # specifically area 2, by combining:
    # 1. Triangles with two vertices on one line and one on an adjacent line (base 2, height 1).
    # 2. Triangles with two vertices on one line and one on a line 2 units away (base 1, height 2).
    # 3. Triangles with one vertex on each line (solving |2x1 - x0 - x2| = 2).
    
    points = []
    for y in range(3):
        for x in range(10):
            points.append((x, y))
            
    return points
