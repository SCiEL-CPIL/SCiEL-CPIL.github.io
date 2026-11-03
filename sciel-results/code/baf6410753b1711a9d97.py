"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Arrange 12 unit hexagons in a compact 3x4 hexagonal grid
    # Flat-top hexagons: horizontal spacing = sqrt(3), vertical = 1.5
    # Odd rows offset by sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    centers = []
    
    # 3 rows, 4 columns
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (row % 2) * (sqrt3 / 2)
            y = row * 1.5
            centers.append((x, y, 0.0))  # angle 0 degrees (flat-top)
    
    # Find the center of mass
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    
    # Translate so center of mass is at origin
    centers = [(x - cx, y - cy, ang) for (x, y, ang) in centers]
    
    # Now find the minimum regular hexagon containing all 12 unit hexagons
    # A regular hexagon of side L centered at (0,0) with rotation theta contains
    # a point (x,y) if the point is within distance L in all 6 normal directions
    
    # For a regular hexagon with side L, the support function in direction u is L * |cos(angle between u and hexagon normal)|
    # Actually, for a regular hexagon with circumradius R = L, the inradius is L * sqrt(3)/2
    
    # A regular hexagon of side L has:
    # - Circumradius = L
    # - Inradius = L * sqrt(3)/2
    # - The hexagon contains a point if the max of the 6 support functions is <= L
    
    # For a hexagon rotated by angle theta, the 6 outward normals are at angles theta + k*pi/3
    # The support function in direction n is: max over vertices of dot(vertex, n)
    
    # For a regular hexagon centered at origin with circumradius R=L, rotated by theta:
    # Vertices are at R * (cos(theta + k*pi/3), sin(theta + k*pi/3)) for k=0..5
    # The hexagon contains point p if for all 6 edge normals n_k, dot(p, n_k) <= inradius = L*sqrt(3)/2
    # Wait, that's not quite right. Let me think again.
    
    # A regular hexagon of side L can be described as the intersection of 6 half-planes.
    # The distance from center to each edge (inradius) is L * sqrt(3)/2.
    # The 6 edge normals are at angles theta + k*pi/3 for k=0..5 (if flat-top) or similar.
    
    # Actually, for a regular hexagon with side length L:
    # - Inradius (apothem) = L * sqrt(3)/2
    # - The hexagon is {p : dot(p, n_k) <= inradius for k=0..5} where n_k are unit normals to edges
    
    # But we also need to account for the fact that each inner hexagon has size (circumradius 1).
    # So the containing hexagon must contain all points within distance 1 of each center.
    # This means: for each inner hexagon center c_i, the entire unit hexagon must be inside.
    # Equivalently, the Minkowski sum of the outer hexagon with a unit disk must contain all centers... no.
    
    # More precisely: outer_hex contains inner_hex_i iff outer_hex contains all points of inner_hex_i.
    # Since inner_hex_i is a unit regular hexagon (circumradius 1), we need:
    # outer_hex contains {c_i + v : v in unit_hexagon}
    # This is equivalent to: (outer_hex - unit_hexagon) contains c_i, where - is Minkowski subtraction.
    # outer_hex - unit_hexagon is a hexagon with "eroded" boundaries.
    
    # For a regular hexagon of side L, subtracting a unit regular hexagon (same orientation) gives
    # a regular hexagon of side L - 1 (if L >= 1). But the inner hexagons can be rotated!
    
    # Since inner hexagons can rotate, the worst case is that we need to contain the convex hull of
    # all rotations, which is a disk of radius 1 (circumradius of unit hexagon).
    # Actually no - the convex hull of all rotations of a regular hexagon is a regular hexagon? No, it's a disk?
    # The convex hull of all rotations of a regular hexagon with circumradius R is a disk of radius R.
    # Wait, no. The regular hexagon has circumradius R=1. The convex hull of all its rotations is the
    # disk of radius 1, since the hexagon contains the inscribed circle of radius sqrt(3)/2, and the
    # convex hull of rotations includes all points at distance up to 1 from center in all directions.
    # Actually, the convex hull of all rotations of a regular hexagon IS a disk of radius equal to the
    # circumradius, because for any direction, there's a vertex in that direction.
    
    # So to be safe, we need the outer hexagon to contain disks of radius 1 around each center.
    # But that's overly conservative. Let me be more precise.
    
    # The condition is: outer_hex ∩ (c_i + rotated_unit_hex) = c_i + rotated_unit_hex for all i and rotations.
    # Since we can choose the rotation of each inner hexagon, we want to find if there exists a rotation
    # for each such that all are contained.
    
    # For simplicity and safety, let's assume inner hexagons are all axis-aligned (angle=0) and
    # compute the required outer hexagon size. We can then optimize the arrangement.
    
    # For axis-aligned inner hexagons (flat-top), the Minkowski sum of two flat-top hexagons of sides L and 1
    # is a flat-top hexagon of side L+1. So we need L+1 >= ... hmm, this is getting complex.
    
    # Let me just do a numerical approach: for a given outer hexagon (center, side, rotation),
    # check if all 12 unit hexagons (with their rotations) are contained and disjoint.
    
    # For now, let's use a reasonable estimate. The bounding box of our 3x4 arrangement:
    # x ranges from -sqrt3/2 to sqrt3/2 + 3*sqrt3 = sqrt3/2 + 3*sqrt3 = 3.5*sqrt3 ≈ 6.06
    # Wait, let me recompute. After centering:
    # x values: for row 0 (even): 0, sqrt3, 2sqrt3, 3sqrt3 -> centered: -1.5sqrt3, -0.5sqrt3, 0.5sqrt3, 1.5sqrt3
    # for row 1 (odd): sqrt3/2, 3sqrt3/2, 5sqrt3/2, 7sqrt3/2 -> centered: -sqrt3, 0, sqrt3, 2sqrt3
    # Hmm, let me just compute numerically.
    
    # Let me try a different approach: use a hexagonal cluster arrangement.
    # 12 hexagons in a hexagonal pattern:
    # Row -2: 2 hexagons
    # Row -1: 3 hexagons  
    # Row 0:  4 hexagons
    # Row 1:  3 hexagons
    # Row 2:  2 hexagons
    # Total: 2+3+4+3+2 = 14, too many.
    
    # Try: 1+4+4+3 = 12? Or 2+3+4+3 = 12?
    # 2+3+4+3 = 12. Let's do that.
    
    centers = []
    # Row 0 (top): 2 hexagons
    # Row 1: 3 hexagons
    # Row 2: 4 hexagons
    # Row 3 (bottom): 3 hexagons
    
    # Using hexagonal coordinates converted to Cartesian
    # For flat-top hexagons in a hex grid:
    # (q, r) -> x = sqrt3 * (q + r/2), y = 1.5 * r
    
    hex_coords = [
        # Row r=-1: 2 hexagons
        (-1, -1), (0, -1),
        # Row r=0: 3 hexagons
        (-1, 0), (0, 0), (1, 0),
        # Row r=1: 4 hexagons
        (-2, 1), (-1, 1), (0, 1), (1, 1),
        # Row r=2: 3 hexagons
        (-1, 2), (0, 2), (1, 2),
    ]
    
    for q, r in hex_coords:
        x = sqrt3 * (q + r / 2)
        y = 1.5 * r
        centers.append((x, y, 0.0))
    
    # Center the arrangement
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy, ang) for (x, y, ang) in centers]
    
    # Now compute the minimum outer hexagon side length
    # We need a regular hexagon (centered at origin, some rotation) that contains all 12 unit hexagons
    # For safety, let's compute the maximum distance from origin to any vertex of any inner hexagon,
    # and then find the hexagon side that can contain that.
    
    # For a unit flat-top hexagon centered at (cx_i, cy_i), the vertices are at:
    # (cx_i + cos(k*pi/3), cy_i + sin(k*pi/3)) for k=0..5 (circumradius 1)
    
    # The outer regular hexagon of side L centered at origin with rotation theta contains
    # a point p if dot(p, n_k) <= L * sqrt(3)/2 for all k=0..5, where n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
    
    # We need this for all vertices of all inner hexagons.
    
    # Let's try theta = 0 and find the minimum L.
    theta = 0.0
    normals = [(np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)) for k in range(6)]
    
    max_proj = 0
    for (cx_i, cy_i, _) in centers:
        for k in range(6):
            vx = cx_i + np.cos(k * np.pi / 3)
            vy = cy_i + np.sin(k * np.pi / 3)
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                max_proj = max(max_proj, proj)
    
    # We need L * sqrt(3)/2 >= max_proj, so L >= max_proj * 2/sqrt(3)
    L = max_proj * 2.0 / sqrt3
    
    # Add a small margin for safety
    L = L * 1.01
    
    return centers, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END

