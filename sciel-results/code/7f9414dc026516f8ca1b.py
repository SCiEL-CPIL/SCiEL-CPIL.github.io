"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # For N=11 unit hexagons, we need to find the minimal outer hexagon side length.
    # The current seed uses a linear arrangement with side=40, which is very poor.
    # A better approach: pack them in a hexagonal cluster pattern.
    
    # Let's think about the geometry.
    # A regular hexagon with side length 1 has a "radius" (center to vertex) of 1.
    # The distance between centers of two touching hexagons is 2 (since each has radius 1 in the direction of contact).
    # Actually, for regular hexagons of side 1:
    # - The distance from center to any vertex is 1.
    # - The distance from center to the middle of any side is sqrt(3)/2.
    # - Two hexagons can touch if the distance between their centers is at least 2 (if aligned vertex-to-vertex) 
    #   or potentially less if rotated. But the minimal distance between centers of two non-overlapping 
    #   unit regular hexagons is 2 (when they share a full edge).
    #   Wait, if they share an edge, the distance between centers is sqrt(3) * 1? No.
    #   Let's re-derive. A unit regular hexagon has side 1. The distance from center to vertex is 1.
    #   If two hexagons share an edge, the distance between centers is the distance from center to the midpoint of an edge, times 2?
    #   No, the distance from center to the midpoint of an edge is apothem = sqrt(3)/2 * 1 = sqrt(3)/2.
    #   So if they share an edge, center-to-center distance is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732.
    #   However, this assumes they are aligned. If rotated, the minimal distance can be different.
    #   But for a simple packing, we can assume aligned hexagons.
    
    # Let's assume all inner hexagons are aligned (angle=0).
    # Then the constraint is: distance between centers >= sqrt(3) if they share an edge,
    # but actually the constraint is more complex: the Minkowski sum of two hexagons.
    # For simplicity, let's use a heuristic packing.
    
    # A hexagonal cluster of N=11:
    # Center: 1
    # First ring: 6
    # Second ring: 4 more (to make 11)
    # Total: 1 + 6 + 4 = 11
    
    # Let's generate positions for a hexagonal grid.
    # Using axial coordinates (q, r).
    # Convert to Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    # But this is for pointy-top or flat-top? Let's use flat-top hexagons.
    # For flat-top hexagons with side 1:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Positions for a hex cluster of radius 1 (which has 7 hexagons) plus 4 more.
    # Let's just place them in a compact shape.
    
    positions = []
    # Center
    positions.append((0, 0))
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * 60
        x = math.cos(math.radians(angle)) * math.sqrt(3)
        y = math.sin(math.radians(angle)) * math.sqrt(3)
        positions.append((x, y))
    # Second ring partial (4 hexagons)
    # Add 4 more from the second ring
    for i in range(4):
        angle = i * 60
        x = math.cos(math.radians(angle)) * 2 * math.sqrt(3)
        y = math.sin(math.radians(angle)) * 2 * math.sqrt(3)
        positions.append((x, y))
    
    # Now we need to find the minimal outer hexagon side length that contains all these.
    # The outer hexagon is also regular. We can rotate it to minimize the side length.
    # For a given set of points, the minimal enclosing regular hexagon can be found by
    # trying different rotations and computing the required side length.
    
    # The side length of the outer hexagon is determined by the maximum distance
    # from the center to any point in the direction of the hexagon's vertices/sides.
    
    # Let's assume the outer hexagon is centered at (0,0) for now (we can shift it later).
    # For a regular hexagon with side length s and rotation theta, the boundary is defined by:
    # max over 6 directions of (x * cos(theta + i*60) + y * sin(theta + i*60)) <= s * sqrt(3) / 2
    # Wait, the apothem of the outer hexagon is s * sqrt(3) / 2.
    # So the constraint is: for all points (x,y) and all 6 normals n_i, n_i . (x,y) <= s * sqrt(3) / 2
    # And also -n_i . (x,y) <= s * sqrt(3) / 2 (since the hexagon is symmetric)
    # So: max_i |n_i . (x,y)| <= s * sqrt(3) / 2
    # Therefore: s >= 2 / sqrt(3) * max_i |n_i . (x,y)|
    
    # We want to minimize s over theta.
    # s(theta) = 2 / sqrt(3) * max_{points p} max_{i=0..5} |n_i(theta) . p|
    
    # Let's compute this for various theta.
    best_s = float('inf')
    best_theta = 0
    for theta_deg in range(0, 60, 1):  # 60 degrees is enough due to symmetry
        theta = math.radians(theta_deg)
        max_val = 0
        for (x, y) in positions:
            for i in range(6):
                angle = theta + i * math.pi / 3
                nx = math.cos(angle)
                ny = math.sin(angle)
                val = abs(nx * x + ny * y)
                if val > max_val:
                    max_val = val
        s = 2 / math.sqrt(3) * max_val
        if s < best_s:
            best_s = s
            best_theta = theta_deg
    
    # Now, we can also try shifting the outer hexagon to better enclose the points.
    # But for now, let's just return this.
    
    # Create the inner list
    inner = [[x, y, 0.0] for (x, y) in positions]
    
    # The outer hexagon center is (0,0), side length is best_s, rotation is best_theta
    return inner, (0.0, 0.0), best_s, best_theta

# EVOLVE-BLOCK-END
