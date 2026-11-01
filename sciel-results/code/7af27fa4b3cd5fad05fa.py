"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 hexagons in a compact cluster
    # Axial coordinates (q, r)
    # Center (0,0)
    # 6 neighbors (distance 1)
    # 4 selected from distance 2 for symmetry
    
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to pixel coordinates (pointy-top, side length 1)
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        # Angle 0.0 means standard orientation
        inner.append([x, y, 0.0])
        
    # Calculate minimal enclosing hexagon side length
    # The cluster is symmetric. The farthest extent is from the center to the farthest vertex.
    # For the hexagon at (2,0), center is at (2*sqrt(3), 0).
    # Its farthest vertex in the +x direction is at (2*sqrt(3) + 1, 0).
    # The outer hexagon (aligned) has a vertex at (L, 0).
    # So L must be at least 2*sqrt(3) + 1.
    
    L = 2 * math.sqrt(3) + 1.0
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
