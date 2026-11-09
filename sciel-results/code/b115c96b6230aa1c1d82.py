"""Improved packing: hexagonal lattice of 11 unit hexagons inside a regular hexagon."""
import time
import math

N = 11

def _axial_to_xy(q, r):
    """Convert axial coordinates to cartesian for pointy-top hexagon of side 1."""
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def _enclosing_hex_side(points):
    """
    Given a list of (x, y) points (centers of unit hexagons, flat-top orientation aligned),
    compute the minimum side length of an enclosing regular hexagon (flat-top, centered at origin,
    rotated to align with the lattice).
    
    For a regular hexagon with side length S, centered at origin, with one vertex pointing up
    (pointy-top), the boundary is:
      |y| <= S
      |x*sqrt(3)/2 + y/2| <= S * sqrt(3)/2  ... actually let me think more carefully.
    
    Actually, for a regular hexagon with side length S:
    - If flat-top (two horizontal edges), the vertices are at angles 0, 60, 120, 180, 240, 300 degrees
      from the positive x-axis, at distance S from center.
      Wait, no. For a regular hexagon with side length S, the circumradius equals S.
      For flat-top orientation, vertices at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
      
    - If pointy-top (two vertices on vertical axis), vertices at (0, S), (S*sqrt(3)/2, S/2), (S*sqrt(3)/2, -S/2), (0, -S), (-S*sqrt(3)/2, -S/2), (-S*sqrt(3)/2, S/2).
    
    The interior of a regular hexagon centered at origin with side S and orientation theta can be described as the intersection of 6 half-planes.
    For a flat-top hexagon (edges at top and bottom are horizontal):
      The normals to the 6 edges point at angles 90, 150, 210, 270, 330, 30 degrees (or 0, 60, 120, 180, 240, 300 for vertex directions).
    
    Let me use the standard: for a regular hexagon with side length S, the inradius (apothem) is S*sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes, each at distance S*sqrt(3)/2 from the center, with normals at angles 0, 60, 120, 180, 240, 300 degrees.
    
    So a point (x,y) is inside iff for all k=0..5:
      x*cos(k*60°) + y*sin(k*60°) <= S*sqrt(3)/2
      x*cos(k*60°) + y*sin(k*60°) >= -S*sqrt(3)/2
    
    Equivalently: max_k |x*cos(k*60°) + y*sin(k*60°)| <= S*sqrt(3)/2
    
    So S = (2/sqrt(3)) * max_k |x*cos(k*60°) + y*sin(k*60°)|
    
    But we also need to account for the fact that the inner hexagons have size. The inner hexagons are unit hexagons with side 1.
    The centers of the inner hexagons must be at least the inradius distance from the boundary... no wait.
    
    The condition is that the entire inner hexagon (with side 1) must be inside the outer hexagon (with side S).
    
    The outer hexagon boundary is at distance S*sqrt(3)/2 from center along each normal direction.
    The inner hexagon, centered at point p, extends in each normal direction by its inradius, which is sqrt(3)/2 (for side 1).
    
    So the condition is: for each of the 6 normal directions n_k,
      (p · n_k) + sqrt(3)/2 <= S*sqrt(3)/2
      (p · n_k) - sqrt(3)/2 >= -S*sqrt(3)/2
    
    This means: |p · n_k| <= S*sqrt(3)/2 - sqrt(3)/2 = (S-1)*sqrt(3)/2
    
    So: max_k |p · n_k| <= (S-1)*sqrt(3)/2
    
    Therefore: S = 1 + (2/sqrt(3)) * max_k |p · n_k|
    """
    max_val = 0.0
    for (px, py) in points:
        for k in range(6):
            angle = k * math.pi / 3.0
            val = abs(px * math.cos(angle) + py * math.sin(angle))
            if val > max_val:
                max_val = val
    return 1.0 + (2.0 / math.sqrt(3.0)) * max_val

def construct():
    start_time = time.time()
    
    # 11 hexagons in a hexagonal cluster: center + 6 neighbors + 4 from next ring
    # Axial coordinates (q, r):
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 (4 more): (2,0), (2,-1), (1,-2), (-1,2), (-2,2), (-2,1), (0,2), (1,1)
    # We need 4 more from ring 2 to make 11 total.
    # Choose symmetrically: (2,0), (0,2), (-2,2), (-2,0) -- wait let me pick 4 that maintain symmetry.
    # Actually, let's use: center(0,0), ring1 all 6, and from ring2 pick 4.
    # For symmetry, pick: (2,0), (0,2), (-2,2), (-2,0)
    # Or better: maintain 6-fold symmetry is impossible with 11 (not 1+6+4=11 but 4 isn't 6).
    # Let's just use a valid compact set.
    
    axial_points = [
        (0, 0),      # center
        (1, 0),      # ring 1
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (2, 0),      # ring 2 - 4 more
        (0, 2),
        (-2, 2),
        (-2, 0),
    ]
    
    # Convert to cartesian
    cartesian_points = [_axial_to_xy(q, r) for q, r in axial_points]
    
    # Calculate enclosing hexagon side
    side = _enclosing_hex_side(cartesian_points)
    
    # Create inner hexagons with positions and angles (all 0 since aligned)
    inner = [[px, py, 0.0] for (px, py) in cartesian_points]
    
    # Outer hexagon centered at origin, side length = computed side, rotation 0
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
