"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    # Honeycomb lattice packing of hexagons
    # For flat-top hexagons with side length s=1:
    # - Width = 2*s = 2
    # - Height = sqrt(3)*s = sqrt(3)
    # - Horizontal spacing between centers in same row = 2*s = 2
    # - Vertical spacing between rows = sqrt(3)*s = sqrt(3)
    # - Odd rows offset by s = 1 horizontally
    
    s = 1.0  # inner hexagon side length
    sqrt3 = math.sqrt(3)
    
    # Honeycomb arrangement: 4 rows of 3 hexagons
    # Row 0 (y=0): x = -2, 0, 2
    # Row 1 (y=sqrt(3)): x = -1, 1, 3  (offset by 1)
    # Row 2 (y=2*sqrt(3)): x = -2, 0, 2
    # Row 3 (y=3*sqrt(3)): x = -1, 1, 3
    
    positions = []
    for row in range(4):
        y = row * sqrt3
        for col in range(3):
            if row % 2 == 0:
                x = (col - 1) * 2.0  # -2, 0, 2
            else:
                x = (col - 1) * 2.0 + 1.0  # -1, 1, 3
            positions.append((x, y))
    
    # All hexagons are flat-top (angle = 0)
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Find bounding box of the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    # For flat-top hexagons with side s:
    # - Leftmost vertex at x - s
    # - Rightmost vertex at x + s
    # - Topmost vertex at y + sqrt(3)/2 * s
    # - Bottommost vertex at y - sqrt(3)/2 * s
    
    min_x = min(xs) - s
    max_x = max(xs) + s
    min_y = min(ys) - sqrt3/2 * s
    max_y = max(ys) + sqrt3/2 * s
    
    # Center of bounding box
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    # The outer hexagon needs to contain all inner hexagons
    # For a regular hexagon with side L centered at (cx, cy) with flat-top orientation:
    # - It contains points within distance L in the "hexagonal metric"
    # - The maximum distance from center to any vertex is L
    # - We need L large enough to contain all inner hexagon vertices
    
    # Calculate the required outer side length
    # For each inner hexagon vertex, compute its distance from outer center
    # and find the maximum "hexagonal radius" needed
    
    def hex_radius_from_center(px, py, cx, cy):
        """Compute the minimum side length of a flat-top hexagon centered at (cx,cy) 
        that contains point (px,py)."""
        dx = px - cx
        dy = py - cy
        # For flat-top hexagon with side L, the boundary is:
        # |dx| <= L, |dy| <= sqrt(3)/2 * L, |dx + dy/sqrt(3)| <= L
        # The hexagonal radius is max(|dx|, |dy|/(sqrt(3)/2), |dx + dy/sqrt(3)|)
        # Actually for flat-top: vertices at angles 0, 60, 120, 180, 240, 300
        # The hexagon is the intersection of 6 half-planes
        # For flat-top hexagon with side L centered at origin:
        # -x <= L, x <= L
        # -sqrt(3)/2*y <= L, sqrt(3)/2*y <= L  
        # Actually let me think more carefully.
        
        # Flat-top regular hexagon with side L, centered at origin:
        # Vertices at: (L,0), (L/2, sqrt(3)L/2), (-L/2, sqrt(3)L/2), (-L,0), (-L/2,-sqrt(3)L/2), (L/2,-sqrt(3)L/2)
        # The hexagon is defined by:
        # |x| <= L
        # |y| <= sqrt(3)*L/2
        # |x + y/sqrt(3)| <= L  -- wait, let me derive properly
        
        # For a flat-top hexagon with "radius" (center to vertex) = L:
        # The six edges are at distances L*cos(30°) = L*sqrt(3)/2 from center
        # Edge normals are at angles 0°, 60°, 120°, 180°, 240°, 300°
        # So the constraints are:
        # For each angle θ in [0°, 60°, 120°, 180°, 240°, 300°]:
        #   x*cos(θ) + y*sin(θ) <= L*sqrt(3)/2
        
        r = 0.0
        for i in range(6):
            theta = i * math.pi / 3
            val = abs(dx * math.cos(theta) + dy * math.sin(theta))
            r = max(r, val)
        # The distance from center to each edge is L*sqrt(3)/2
        # So L = r / (sqrt(3)/2) = 2*r/sqrt(3)
        return 2.0 * r / sqrt3
    
    # Check all vertices of all inner hexagons
    max_L = 0.0
    for x, y, ang in inner:
        # Get vertices of this inner hexagon
        for i in range(6):
            theta = math.radians(ang) + i * math.pi / 3
            vx = x + s * math.cos(theta)
            vy = y + s * math.sin(theta)
            L_needed = hex_radius_from_center(vx, vy, cx, cy)
            max_L = max(max_L, L_needed)
    
    outer_side = max_L * 1.001  # small margin for floating point
    
    return inner, (cx, cy), outer_side, 0.0
# EVOLVE-BLOCK-END
