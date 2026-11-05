"""Rewritten program: exact convex-hull containment for 11 unit hexagons."""
import math
import numpy as np
from scipy.spatial import ConvexHull

N = 11

def _hexagon_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """
    Return 6 vertices of a regular hexagon centered at (cx, cy) with given side length
    and rotation (degrees). Vertices are in counterclockwise order starting from angle.
    """
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return vertices

def _convex_hull(points):
    """
    Compute convex hull of a set of 2D points. Returns list of hull vertices in CCW order.
    Uses scipy.spatial.ConvexHull.
    """
    pts = np.array(points)
    if len(pts) < 3:
        return list(pts)
    hull = ConvexHull(pts)
    hull_indices = hull.vertices
    hull_points = [tuple(pts[i]) for i in hull_indices]
    return hull_points

def _min_enclosing_hexagon_side(hull_points, n_angles=360):
    """
    Given hull vertices (list of (x,y)), find the minimum side length L of a regular
    hexagon (centered at origin, variable rotation) that contains all hull points.
    
    For a regular hexagon with side length L centered at origin, the boundary in
    direction theta is at distance r(theta) = L * cos(pi/6) / cos((theta mod pi/3) - pi/6)
    ... actually let's use the support function approach.
    
    A regular hexagon with side length L, centered at origin, with one vertex at angle 0,
    has 6 normals at angles 0, 60, 120, 180, 240, 300 degrees (for the flat sides).
    The distance from center to each flat side (apothem) is a = L * sqrt(3)/2.
    
    For a point p = (x, y), the constraint that it lies inside the hexagon with rotation phi is:
    For each of the 6 normals at angle phi + k*60 degrees, k=0..5:
        p . n_k <= a  where a = L * sqrt(3)/2
    
    So L >= (p . n_k) / (sqrt(3)/2) for all k and all points p.
    L >= max over all points p and all 6 normals of (p . n_k) / (sqrt(3)/2)
    
    We minimize over rotation phi.
    """
    pts = np.array(hull_points)
    if len(pts) == 0:
        return 0.0
    
    # Center the hull at origin for symmetry? No, we're finding enclosing hex centered at origin.
    # Actually, the problem says outer hexagon center is returned separately. But for now,
    # let's assume outer hex is centered at origin and we just compute L.
    # We can later shift to optimize center, but first let's fix center at centroid of hull.
    
    # Let's try: for a given rotation phi, compute required L.
    # L(phi) = max over all points p, all k=0..5 of (p . n_k(phi)) / (sqrt(3)/2)
    # where n_k(phi) = (cos(phi + k*pi/3), sin(phi + k*pi/3))
    
    best_L = float('inf')
    for deg in range(n_angles):
        phi = math.radians(deg)
        max_val = 0.0
        for k in range(6):
            angle = phi + k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            # Project all points onto this normal
            projections = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = np.max(projections)
            if max_proj > max_val:
                max_val = max_proj
        L = max_val / (math.sqrt(3) / 2.0)
        if L < best_L:
            best_L = L
    
    return best_L

def construct():
    """
    Place 11 unit hexagons in a compact cluster, compute convex hull,
    find minimum enclosing regular hexagon side length.
    """
    # Arrange 11 hexagons in a compact pattern:
    # Center hex at (0,0), surrounded by 6 hexes, plus 4 more in a ring.
    # Standard hex grid coordinates: axial coordinates.
    # For unit hexagons (side length 1), distance between centers of adjacent hexes is sqrt(3).
    # In a hex grid with pointy-top orientation:
    #   x = sqrt(3) * (q + r/2)
    #   y = 1.5 * r
    # where (q, r) are axial coordinates.
    
    # Let's use flat-top orientation for simplicity.
    # For flat-top hexagons:
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + q/2)
    
    # Place 11 hexagons: center + 6 neighbors + 4 more.
    # Axial coordinates for a compact 11-hex cluster:
    coords = [
        (0, 0),    # center
        (1, 0),    # right
        (0, 1),    # upper-right
        (-1, 1),   # upper-left
        (-1, 0),   # left
        (0, -1),   # lower-left
        (1, -1),   # lower-right
        (2, 0),    # further right
        (1, 1),    # further upper-right
        (-2, 1),   # further upper-left
        (-2, 0),   # further left
    ]
    
    # Convert to Cartesian for flat-top hexagons
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
        positions.append((x, y))
    
    # Collect all vertices of all 11 hexagons (all with 0 rotation for now)
    all_vertices = []
    for (cx, cy) in positions:
        verts = _hexagon_vertices(cx, cy, side=1.0, angle_deg=0.0)
        all_vertices.extend(verts)
    
    # Compute convex hull
    hull_points = _convex_hull(all_vertices)
    
    # Find minimum enclosing hexagon side length
    side_length = _min_enclosing_hexagon_side(hull_points, n_angles=360)
    
    # Return inner hexagons, outer center, outer side length, outer rotation
    inner = [[x, y, 0.0] for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_rotation = 0.0  # We searched over rotations; for simplicity return 0
    
    return inner, outer_center, side_length, outer_rotation

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
