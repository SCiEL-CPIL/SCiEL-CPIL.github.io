"""Program for packing 11 unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.pi / 3 * i
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _convex_hull(points):
    """Compute convex hull of a set of 2D points using Andrew's monotone chain."""
    points = np.array(points, dtype=float)
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points
    points = np.sort(points, axis=0)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.vstack((lower[:-1], upper[:-1]))

def _cluster_points():
    """
    Generate the 11 unit hexagons in a compact arrangement.
    Using a linear chain along x-axis: positions at (4k, 0) for k in range(-5, 6).
    All with angle 0.
    """
    positions = []
    for k in range(-(N // 2), N - N // 2):
        positions.append((4.0 * k, 0.0, 0.0))
    return positions

def _all_vertices(positions, global_angle_deg=0.0):
    """
    Get all vertices of all hexagons, rotated by global_angle_deg around origin.
    Each hexagon has side length 1.
    """
    all_verts = []
    for (cx, cy, ang_deg) in positions:
        # Local vertices of unit hexagon
        local_verts = _hex_vertices(cx, cy, 1.0, ang_deg)
        # Rotate by global angle
        a = np.radians(global_angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        for v in local_verts:
            rx = cos_a * v[0] - sin_a * v[1]
            ry = sin_a * v[0] + cos_a * v[1]
            all_verts.append((rx, ry))
    return np.array(all_verts)

def _min_enclosing_hex_side(hull_pts):
    """
    Given convex hull points, find the minimum side length of a regular hexagon
    (centered at origin, variable rotation) that encloses all points.
    
    For a regular hexagon centered at origin with side s and rotation phi,
    the constraint is that for each vertex v of the inner cluster,
    v must satisfy: max over 6 directions of dot(v, n_i) <= s * sqrt(3)/2 * ... 
    
    Actually, for a regular hexagon with side length s, the distance from center
    to each edge (apothem) is s * sqrt(3)/2.
    The hexagon is the intersection of 6 half-planes.
    For a hexagon with rotation phi, the 6 outward normals are at angles phi + i*pi/3.
    A point p is inside if for all i: dot(p, n_i) <= apothem = s * sqrt(3)/2.
    
    So for a given rotation phi, the minimum s is:
    s = (2 / sqrt(3)) * max_i max_p dot(p, n_i(phi + i*pi/3))
    
    We want to minimize over phi.
    """
    def required_side(phi):
        apothem_factors = []
        for i in range(6):
            angle = phi + i * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            dots = hull_pts[:, 0] * nx + hull_pts[:, 1] * ny
            max_dot = np.max(dots)
            apothem_factors.append(max_dot)
        max_apothem = max(apothem_factors)
        # s * sqrt(3)/2 >= max_apothem => s >= 2 * max_apothem / sqrt(3)
        return 2.0 * max_apothem / np.sqrt(3.0)
    
    return required_side

def construct():
    positions = _cluster_points()
    
    # Compute convex hull of the cluster (at global angle 0)
    # But we need to consider rotation, so let's work with the full set of vertices
    # For a given global rotation, the hull changes. However, the set of vertices
    # is just a rotation of the original set, so the hull is the rotation of the original hull.
    
    # Let's compute the hull at angle 0
    verts_at_0 = _all_vertices(positions, 0.0)
    hull = _convex_hull(verts_at_0)
    
    # Now, for a global rotation phi, the hull is just hull rotated by phi.
    # The function required_side(phi) depends on the rotated hull.
    
    def side_length(phi_deg):
        phi = np.radians(phi_deg)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        # Rotate hull
        hull_rot = np.zeros_like(hull)
        hull_rot[:, 0] = cos_p * hull[:, 0] - sin_p * hull[:, 1]
        hull_rot[:, 1] = sin_p * hull[:, 0] + cos_p * hull[:, 1]
        return _min_enclosing_hex_side(hull_rot)(0.0)  # phi=0 for the enclosing hexagon's rotation relative to rotated hull
    
    # Wait, I need to be more careful. The enclosing hexagon also has a rotation.
    # Let me redefine: we fix the cluster rotation at theta, and find the best enclosing hexagon rotation.
    # But by symmetry, we can fix the enclosing hexagon orientation and rotate the cluster.
    
    # Let's fix the enclosing hexagon at rotation 0, and rotate the cluster by theta.
    # Then minimize over theta in [0, 60).
    
    def objective(theta_deg):
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate hull by theta
        hull_rot = np.zeros_like(hull)
        hull_rot[:, 0] = cos_t * hull[:, 0] - sin_t * hull[:, 1]
        hull_rot[:, 1] = sin_t * hull[:, 0] + cos_t * hull[:, 1]
        # Enclosing hexagon at rotation 0: normals at 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
        # Apothem required = max over 6 directions of max dot product
        max_apothem = 0.0
        for i in range(6):
            angle = i * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            dots = hull_rot[:, 0] * nx + hull_rot[:, 1] * ny
            max_apothem = max(max_apothem, np.max(dots))
        return 2.0 * max_apothem / np.sqrt(3.0)
    
    # Minimize over theta in [0, 60)
    result = minimize_scalar(objective, bounds=(0, 60), method='bounded',
                             options={'xatol': 1e-10})
    
    best_theta = result.x
    best_side = result.fun
    
    # The cluster is rotated by best_theta, enclosing hexagon at rotation 0
    # We need to output the inner hexagons with their absolute angles and positions
    # and the outer hexagon parameters
    
    # Inner hexagons: each was at (cx, cy, ang_deg) in local frame
    # After global rotation by best_theta, their positions and angles change
    theta_rad = np.radians(best_theta)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    
    inner = []
    for (cx, cy, ang_deg) in positions:
        # Rotate position
        rx = cos_t * cx - sin_t * cy
        ry = sin_t * cx + cos_t * cy
        # Rotate angle
        new_ang = (ang_deg + best_theta) % 360.0
        inner.append([rx, ry, new_ang])
    
    # Outer hexagon: center (0,0), side = best_side, rotation = 0
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
