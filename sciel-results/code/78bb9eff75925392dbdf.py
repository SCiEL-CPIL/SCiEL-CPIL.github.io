"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin
    
    # Axial coordinates for a side-3 hexagon centered at origin
    # |q|<=2, |r|<=2, |q+r|<=2
    cells = []
    for r in range(-2, 3):
        for q in range(-2, 3):
            if abs(q + r) <= 2:
                cells.append((q, r))
    
    # We need to place 11 unit hexagons on 11 of these 19 cells
    # All hexagons have the same orientation (0 degrees) for optimal grid packing
    # The outer hexagon has side length 3.0
    
    # Convert axial to Cartesian coordinates
    # For a regular hexagon with side length 1, the distance between adjacent centers is sqrt(3)
    # Axial to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    sqrt3 = np.sqrt(3.0)
    
    # Pick 11 cells - use a compact arrangement
    # Let's select cells to form a compact cluster
    # We'll use the first 11 cells in a systematic order
    selected = cells[:N]
    
    inner = []
    for (q, r) in selected:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        angle = 0.0
        inner.append([x, y, angle])
    
    center = (0.0, 0.0)
    side = 3.0
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
# EVOLVE-BLOCK-END
