"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate a dense hexagonal lattice of unit hexagons
    # Hexagonal tiling of unit hexagons (side length 1)
    # Distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    # In a rotated coordinate system where x is horizontal:
    # dx = 1.5 (for next column)
    # dy = sqrt(3) (for next row in same column)
    # For column j, x_j = 1.5 * j
    # For column j, y_i = sqrt(3) * (i + 0.5 * (j % 2))
    
    # We want to place N hexagons.
    # Let's try to form a compact cluster.
    # A hexagonal number H_k = 3k(k-1)+1
    # H_1 = 1
    # H_2 = 7
    # H_3 = 19
    # Since N=11, a full H_2 (7) plus 4 more is a good candidate for compactness.
    # Or just take the first N points from a spiral or row-major traversal of a large grid.
    
    # Let's generate a grid of indices (i, j) and map them to coordinates.
    # We'll pick a subset that is roughly circular to minimize the bounding hexagon.
    
    # Generate a large enough grid
    max_idx = 10
    candidates = []
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = 1.5 * j
            y = math.sqrt(3) * (i + 0.5 * (j % 2))
            # Distance from origin
            dist = math.sqrt(x*x + y*y)
            candidates.append((dist, x, y, i, j))
            
    # Sort by distance from origin to pick the most central, compact cluster
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N candidates
    # Ensure we don't pick duplicates if any (shouldn't happen with unique i,j)
    selected = candidates[:N]
    
    # Construct inner hexagon positions
    # The task requires: inner rows (x, y, angle_degrees)
    # All inner hexagons can have angle 0.0
    inner = []
    for dist, x, y, i, j in selected:
        inner.append([x, y, 0.0])
        
    # 2. Calculate the minimal outer hexagon side length
    # The outer hexagon is centered at (0,0) and rotated by 0.0 (aligned with axes).
    # A regular hexagon with side length S and rotation 0 has vertices at:
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # Wait, standard orientation:
    # If "flat top" or "pointy top"?
    # Usually "regular hexagon" in these tasks implies vertices at angles 0, 60, 120...
    # Let's assume the outer hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # The distance from center to vertex is R = S.
    # The distance from center to edge (apothem) is A = S * sqrt(3) / 2.
    
    # For a point (x, y) to be inside the hexagon, it must satisfy:
    # |x| <= A (for vertical sides? No, depends on orientation)
    
    # Let's define the hexagon by its 6 supporting lines.
    # If vertices are at 0, 60, ..., 300:
    # The edges are perpendicular to the bisectors of the angles.
    # Bisectors are at 30, 90, 150, 210, 270, 330.
    # The distance from origin to each edge is A = S * sqrt(3) / 2.
    # Equation of line with normal angle theta: x cos(theta) + y sin(theta) <= A.
    # So for each point (x,y), we need S * sqrt(3) / 2 >= max_{theta in {30, 90, ...}} (x cos(theta) + y sin(theta)).
    # Therefore, S >= 2 / sqrt(3) * max_{theta} (x cos(theta) + y sin(theta)).
    
    thetas = [math.radians(30 + 60 * k) for k in range(6)]
    cos_thetas = [math.cos(t) for t in thetas]
    sin_thetas = [math.sin(t) for t in thetas]
    
    max_projection = 0.0
    for x, y, _ in inner:
        for c, s in zip(cos_thetas, sin_thetas):
            proj = x * c + y * s
            if proj > max_projection:
                max_projection = proj
                
    # The side length S must satisfy S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    side_length = 2.0 * max_projection / math.sqrt(3.0)
    
    # Add a tiny epsilon to ensure strict disjointness/containment if needed, 
    # though usually "inside" allows touching. The evaluator might require strict < or <=.
    # Given the lattice spacing is exact sqrt(3), touching is expected.
    # Let's add a very small buffer to be safe against floating point errors.
    side_length += 1e-9
    
    # Return interface: (inner, center, side, angle)
    # inner: list of [x, y, angle]
    # center: (x, y)
    # side: float
    # angle: float (rotation of outer hexagon)
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
