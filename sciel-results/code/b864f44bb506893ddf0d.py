"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 points from a hexagonal lattice cluster of radius 2 (19 points total)
    # We use axial coordinates (q, r)
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian
                # Lattice spacing is 2.0
                x = 2.0 * q + r
                y = np.sqrt(3) * r
                points.append((x, y))
    
    # We have 19 points. We need 12.
    # Sort by distance from origin to keep the cluster compact
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12
    selected = points[:N]
    
    inner = []
    for x, y in selected:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length 5.0 is sufficient for this packing
    center = (0.0, 0.0)
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
