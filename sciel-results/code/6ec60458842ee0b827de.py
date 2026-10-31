import numpy as np
import time

N = 11

def construct():
    # Hexagonal lattice basis
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 3/2)
    
    # Lattice coordinates for 11 hexagons
    # 1 center, 6 neighbors, 4 from next ring
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # (2,0), (-2,0), (0,2), (0,-2)
    
    lattice_points = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    inner = []
    for i, j in lattice_points:
        x = np.sqrt(3) * i + (np.sqrt(3) / 2) * j
        y = (3 / 2) * j
        angle = 0.0
        inner.append([x, y, angle])
        
    center = (0.0, 0.0)
    side = 5.0
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
