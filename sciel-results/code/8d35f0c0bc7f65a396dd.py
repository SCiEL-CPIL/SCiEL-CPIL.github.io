import numpy as np
from scipy.spatial import ConvexHull
import math

N = 11

def construct():
    # Define 11 unit hexagon centers in a compact cluster
    # Using hexagonal lattice: distance between centers = sqrt(3)
    sqrt3 = math.sqrt(3)
    
    # Compact arrangement: rows with offsets
    # Center positions for 11 hexagons
    centers = []
    # Row 0: 3 hexagons
    for i in range(3):
        centers.append((i * sqrt3 - sqrt3, 0.0))
    # Row 1: 4 hexagons (offset by sqrt3/2, y = 3/2)
    for i in range(4):
        centers.append((i * sqrt3 - 3*sqrt3/2 + sqrt3/2, 3.0/2))
    # Row 2: 3 hexagons
    for i in range(3):
        centers.append((i * sqrt3 - sqrt3, 3.0))
    # Row 3: 1 hexagon
    centers.append((0.0, 9.0/2))
    
    # Generate all vertices of all 11 hexagons
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = math.radians(60 * k)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # Compute convex hull
    hull = ConvexHull(all_vertices)
    hull_points = all_vertices[hull.vertices]
    
    def min_enclosing_hex_side(theta):
        """Compute minimal side length of enclosing regular hexagon at rotation theta."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Rotate hull points
        rotated = np.column_stack([
            hull_points[:, 0] * cos_t - hull_points[:, 1] * sin_t,
            hull_points[:, 0] * sin_t + hull_points[:, 1] * cos_t
        ])
        
        # A regular hexagon with side length s and rotation 0 has vertices at
        # s * (cos(60k), sin(60k)) for k=0..5, centered at origin.
        # The hexagon is the intersection of 6 half-planes.
        # For a hexagon with side s centered at origin, the support function in direction (cos(a), sin(a)) is s * cos(a - 60k) for the appropriate face.
        # Actually, the regular hexagon with side s centered at origin, with one vertex at angle 0, has apothem = s * sqrt(3)/2.
        # The 6 face normals are at angles 0, 60, 120, 180, 240, 300 degrees.
        # The distance from center to each face (apothem) is s * sqrt(3)/2.
        
        # For the hexagon to contain the rotated points, for each of the 6 face directions,
        # the max projection of rotated points onto that direction must be <= apothem = s * sqrt(3)/2.
        # So s >= max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3)
        
        max_s = 0.0
        for k in range(6):
            face_angle = math.radians(60 * k)
            nx = math.cos(face_angle)
            ny = math.sin(face_angle)
            projections = rotated[:, 0] * nx + rotated[:, 1] * ny
            max_proj = np.max(projections)
            # The hexagon centered at origin with this face orientation needs apothem >= max_proj
            # apothem = s * sqrt(3)/2, so s >= 2 * max_proj / sqrt(3)
            s_needed = 2.0 * max_proj / sqrt3
            if s_needed > max_s:
                max_s = s_needed
        
        return max_s
    
    # Ternary search over theta in [0, 2*pi/3) due to hexagon symmetry
    # Actually, the cluster has some symmetry, but let's search [0, pi/3) for safety
    lo = 0.0
    hi = math.pi / 3.0
    for _ in range(100):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = min_enclosing_hex_side(m1)
        f2 = min_enclosing_hex_side(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    
    theta_opt = (lo + hi) / 2.0
    side_opt = min_enclosing_hex_side(theta_opt)
    
    # Rotate all hexagon centers by theta_opt
    cos_t = math.cos(theta_opt)
    sin_t = math.sin(theta_opt)
    rotated_centers = []
    for cx, cy in centers:
        rx = cx * cos_t - cy * sin_t
        ry = cx * sin_t + cy * cos_t
        rotated_centers.append([rx, ry, 0.0])
    
    return rotated_centers, (0.0, 0.0), side_opt, 0.0

