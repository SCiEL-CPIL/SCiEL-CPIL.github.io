"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def get_convex_hull(points):
    """Compute convex hull of a set of points."""
    hull = ConvexHull(points)
    return points[hull.vertices]

def min_enclosing_hexagon_side(hull_points):
    """
    Find the minimum side length of a regular hexagon that encloses the given hull points.
    The center of the enclosing hexagon is the centroid of the hull points for simplicity,
    but we can also optimize the center. For regular hexagons, the minimum enclosing one
    is often centered at the centroid or a vertex/edge midpoint.
    We sweep the angle of the enclosing hexagon.
    """
    # Centroid of the hull points as a good guess for the center
    cx = np.mean(hull_points[:, 0])
    cy = np.mean(hull_points[:, 1])
    
    min_side = float('inf')
    best_angle = 0.0
    
    # The symmetry of the regular hexagon means we only need to check angles 0 to 60 degrees.
    # We sample angles finely.
    angles = np.linspace(0, 60, 121)
    
    for theta_deg in angles:
        theta_rad = np.radians(theta_deg)
        # Normals to the sides of the enclosing hexagon
        # A regular hexagon with "flat top" (vertex at 0 deg) has normals at 0, 60, ...
        # Let's define the hexagon by its 6 supporting lines.
        # If the hexagon is rotated by theta, the normals are at theta, theta+60, ...
        normals = []
        for k in range(6):
            angle = theta_rad + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        # For a hexagon centered at (cx, cy) with side length s,
        # the distance from center to each side (apothem) is s * sqrt(3) / 2.
        # The projection of any point P onto a normal n must be <= apothem + projection of center onto n?
        # Actually, the condition for a point P to be inside a hexagon centered at C with normals N and apothem A is:
        # dot(P - C, N_i) <= A for all i.
        # So A >= max_i dot(P - C, N_i) for all P.
        # Thus A_min = max_{P in hull, i in 0..5} dot(P - C, N_i)
        # And s = A_min * 2 / sqrt(3)
        
        # Calculate max projection for all points and all normals
        # P - C
        diffs = hull_points - np.array([cx, cy])
        
        max_proj = 0.0
        for n in normals:
            projs = np.dot(diffs, n)
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
        
        # Ensure non-negative (if center is outside, this might fail, but centroid is inside convex hull)
        if max_proj < 0:
            max_proj = 0.0
            
        side = max_proj * 2.0 / np.sqrt(3)
        
        if side < min_side:
            min_side = side
            best_angle = theta_deg
            
    return min_side, best_angle, (cx, cy)

def construct():
    """Construct 12 unit hexagons inside a minimal outer hexagon."""
    
    # Define 12 positions in a hexagonal lattice (axial coordinates)
    # Shape: A compact cluster.
    # Axial coords (q, r)
    # This specific set of 12 forms a shape that is 3 units wide and roughly circular.
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, -1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, -1), (-1, 1)
    ]
    
    # Convert to Cartesian coordinates for flat-top hexagons
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    positions = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        positions.append((x, y, 0.0)) # Angle 0 for all (aligned)
        
    # Collect all vertices of the 12 inner hexagons
    all_vertices = []
    for x, y, angle in positions:
        verts = hex_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
        
    all_vertices = np.vstack(all_vertices)
    
    # Compute convex hull of the union
    hull_points = get_convex_hull(all_vertices)
    
    # Find minimum enclosing regular hexagon
    side, angle_deg, (cx, cy) = min_enclosing_hexagon_side(hull_points)
    
    # The inner hexagons are at the calculated positions with angle 0
    inner = [[p[0], p[1], p[2]] for p in positions]
    
    # Return format: inner, center, side, angle
    # center is (cx, cy)
    # side is the side length
    # angle is the rotation of the outer hexagon
    return inner, (cx, cy), side, angle_deg

# EVOLVE-BLOCK-END
