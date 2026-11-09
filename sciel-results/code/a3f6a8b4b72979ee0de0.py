"""Optimized packing of 12 unit regular hexagons inside a regular hexagon."""
import numpy as np
from itertools import product

N = 12

def construct():
    """
    Pack 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy: Use a grid-like arrangement with slight perturbations and rotations
    to achieve a tighter packing than the trivial linear arrangement.
    
    The seed program places hexagons in a line with side length 40.0, which is very
    loose. A better arrangement uses the natural hexagonal lattice structure.
    
    For 12 hexagons, a good arrangement is a central hexagon surrounded by 6,
    then 5 more around the outside, or a 3x4 rectangular-like grid in hexagonal
    coordinates.
    
    We'll use a hexagonal lattice arrangement where all inner hexagons are axis-aligned
    (0 degree rotation) and placed at hexagonal grid positions.
    """
    # Hexagonal lattice: place hexagons in a pattern that minimizes the bounding hexagon
    # 
    # A regular hexagon with side length s can contain unit hexagons arranged in
    # a hexagonal grid. The distance between centers of adjacent unit hexagons
    # in a hexagonal lattice is sqrt(3) (for side length 1 hexagons).
    #
    # For 12 hexagons, let's try a 3-4-3 or 2-3-3-2-2 arrangement, or a 
    # rectangular-ish grid in hex coordinates.
    #
    # Actually, let's think about this more carefully.
    #
    # A unit regular hexagon has inradius (apothem) = sqrt(3)/2 ≈ 0.866
    # and circumradius = 1.
    #
    # If we place unit hexagons on a hexagonal lattice with spacing d between centers,
    # they don't overlap if d >= sqrt(3) (the distance between parallel sides).
    #
    # For the outer hexagon of side L, we need all inner hexagons to fit inside.
    # The constraint is that for each inner hexagon center (x,y), the hexagon must
    # be contained within the outer hexagon.
    #
    # Let's try a simple approach: place hexagons in a 3x4 grid in axial coordinates,
    # then compute the minimum enclosing hexagon.
    
    # Hexagonal lattice positions for 12 hexagons
    # Using axial coordinates (q, r) and converting to Cartesian
    # In a hex grid, adjacent hexagons have centers at distance sqrt(3) apart
    # for unit hexagons (side length 1).
    
    # Let's use a compact arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # Total: 12
    
    # Actually, let me think about what minimizes the outer hexagon side.
    # The outer hexagon should be oriented the same as the inner ones for symmetry.
    
    # For a regular hexagon of side L centered at origin, a point (x,y) is inside
    # if:
    # |x| <= L * cos(0) = L  ... no, that's not right.
    # 
    # A regular hexagon with side L, centered at origin, with flat top (point at angle 0):
    # Vertices at (L*cos(k*60), L*sin(k*60)) for k=0..5
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
    # The constraints are:
    # x * cos(theta) + y * sin(theta) <= L * cos(30) = L * sqrt(3)/2
    # for theta = 0, 60, 120, 180, 240, 300
    # Wait, let me reconsider.
    #
    # A regular hexagon with side length L has apothem (inradius) = L * sqrt(3)/2.
    # If the hexagon is oriented with a flat side on top (vertex at angle 90°),
    # or with a vertex on the right (vertex at angle 0°).
    #
    # Let's use the orientation where vertices are at angles 0°, 60°, 120°, 180°, 240°, 300°.
    # Then the 6 supporting lines are at angles 30°, 90°, 150°, 210°, 270°, 330° from the x-axis.
    # The distance from center to each side is the apothem = L * sqrt(3)/2.
    #
    # So a point (x,y) is inside if for all k in 0..5:
    # x * cos(30° + k*60°) + y * sin(30° + k*60°) <= L * sqrt(3)/2
    
    # For each inner hexagon of side 1 at position (cx, cy) with rotation angle a,
    # we need all 6 vertices of the inner hexagon to be inside the outer hexagon.
    # Actually, since both are convex, it's sufficient to check that all vertices
    # of the inner hexagon are inside the outer hexagon.
    
    # Let's try a few arrangements and compute the required L for each.
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def min_outer_side(inner_positions, inner_angles, outer_angle_deg=0.0):
        """Compute minimum outer hexagon side length for given inner hexagons."""
        outer_angle_rad = np.radians(outer_angle_deg)
        # The 6 normal directions for the outer hexagon
        normals = []
        for k in range(6):
            theta = outer_angle_rad + np.pi / 6 + k * np.pi / 3
            normals.append((np.cos(theta), np.sin(theta)))
        
        max_projection = 0
        for (cx, cy), angle in zip(inner_positions, inner_angles):
            verts = hex_vertices(cx, cy, 1.0, angle)
            for (vx, vy) in verts:
                for (nx, ny) in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_projection:
                        max_projection = proj
        
        # The apothem of the outer hexagon must be at least max_projection
        # Apothem = L * sqrt(3)/2, so L = 2 * max_projection / sqrt(3)
        return 2.0 * max_projection / np.sqrt(3)
    
    # Try arrangement 1: 3 rows of 4
    # Hex grid with spacing sqrt(3) between centers
    d = np.sqrt(3)
    positions = []
    angles = []
    for row in range(3):
        for col in range(4):
            x = col * d
            y = row * d * 1.5  # vertical spacing for hex grid
            if row % 2 == 1:
                x += d / 2
            positions.append((x, y))
            angles.append(0.0)
    
    # Center the arrangement
    cx_mean = np.mean([p[0] for p in positions])
    cy_mean = np.mean([p[1] for p in positions])
    positions = [(x - cx_mean, y - cy_mean) for (x, y) in positions]
    
    L1 = min_outer_side(positions, angles)
    
    # Try arrangement 2: 2-3-3-2-2 pattern (more compact)
    positions2 = []
    angles2 = []
    # Row 0: 2 hexagons
    for col in range(2):
        positions2.append((col * d - d/2, 0))
        angles2.append(0.0)
    # Row 1: 3 hexagons
    for col in range(3):
        positions2.append((col * d - d, d * 1.5))
        angles2.append(0.0)
    # Row 2: 3 hexagons
    for col in range(3):
        positions2.append((col * d - d, -d * 1.5))
        angles2.append(0.0)
    # Row 3: 2 hexagons
    for col in range(2):
        positions2.append((col * d - d/2, -d * 3.0))
        angles2.append(0.0)
    # Row 4: 2 hexagons
    for col in range(2):
        positions2.append((col * d - d/2, d * 3.0))
        angles2.append(0.0)
    
    cx_mean2 = np.mean([p[0] for p in positions2])
    cy_mean2 = np.mean([p[1] for p in positions2])
    positions2 = [(x - cx_mean2, y - cy_mean2) for (x, y) in positions2]
    
    L2 = min_outer_side(positions2, angles2)
    
    # Try arrangement 3: central hexagon + ring of 6 + 5 more
    positions3 = [(0, 0)]
    angles3 = [0.0]
    # Ring of 6 at distance d
    for k in range(6):
        theta = k * np.pi / 3
        positions3.append((d * np.cos(theta), d * np.sin(theta)))
        angles3.append(0.0)
    # 5 more around the outside
    # Place them at the outer positions
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 10
        r = d * 2
        positions3.append((r * np.cos(theta), r * np.sin(theta)))
        angles3.append(0.0)
    
    cx_mean3 = np.mean([p[0] for p in positions3])
    cy_mean3 = np.mean([p[1] for p in positions3])
    positions3 = [(x - cx_mean3, y - cy_mean3) for (x, y) in positions3]
    
    L3 = min_outer_side(positions3, angles3)
    
    # Pick the best arrangement
    best_L = min(L1, L2, L3)
    if L1 == best_L:
        best_positions, best_angles = positions, angles
    elif L2 == best_L:
        best_positions, best_angles = positions2, angles2
    else:
        best_positions, best_angles = positions3, angles3
    
    # Convert to the required format: [x, y, angle_degrees]
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(best_positions, best_angles)]
    
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-START
# (The construct function is defined above, so we just need to keep this block structure)
# EVOLVE-BLOCK-END
