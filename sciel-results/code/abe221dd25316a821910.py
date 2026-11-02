"""Honeycomb Lattice Deformation: place 12 unit hexagons in a 3x4 sub-lattice,
optimize global shear + rotation to minimize the bounding regular hexagon side length."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, side, rot_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side length `side`, rotation in degrees."""
    r = np.radians(rot_deg)
    c, s = np.cos(r), np.sin(r)
    # standard hex vertices (pointy-top at 0 deg): angles 0,60,...,300
    ang = np.deg2rad(np.arange(0, 360, 60))
    pts = side * np.column_stack([np.cos(ang), np.sin(ang)])
    # rotate
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts + np.array([cx, cy])

def _max_radius_hexagon(points):
    """Minimum side length of an axis-aligned (rot=0) regular hexagon centered at origin
    that contains all `points`. Side length = max over 6 support directions of (max dot).
    For a regular hexagon centered at origin with side s and rotation 0 (pointy-top),
    the half-width in direction u is s * |cos(angle between u and normal)| ... 
    Actually: regular hexagon = intersection of 6 half-planes. With side s, apothem a = s*sqrt(3)/2.
    The 6 outward normals are at angles 0,60,...,300 (for flat-top) or 30,90,... for pointy.
    Let's use: hexagon with side s, rotation theta. The support function h(u) = a * max_k cos(angle(u - n_k))
    where n_k are the 6 face-normal directions. A point p is inside iff for all k, p·n_k <= a.
    So min s such that all points inside: for a given theta, a >= max_p max_k p·n_k(theta).
    s = 2*a/sqrt(3).
    """
    # We'll compute for a given rotation of the OUTER hexagon, the required side.
    # But here we want: given points, find min over outer rotation of the enclosing side.
    # That's a 1D problem. We'll handle rotation externally.
    # This helper: given points and outer rotation, return required side length.
    raise NotImplementedError("use _enclosing_side(points, rot_deg)")

def _enclosing_side(points, rot_deg):
    """Given a set of 2D points, return the minimum side length of a regular hexagon
    with rotation rot_deg (degrees) and center at origin that contains all points.
    If no such hexagon centered at origin exists (i.e., points not centered), this
    still returns the value assuming center=origin; caller should center points first.
    """
    r = np.radians(rot_deg)
    # Face normals of a regular hexagon with rotation rot_deg.
    # For a regular hexagon, the 6 face normals are equally spaced by 60 degrees.
    # The orientation: if rot_deg=0 means "pointy-top" (vertex at top), then face normals
    # are at 30, 90, 150, 210, 270, 330 degrees.
    # Let's define: at rot_deg=0, the hexagon has a flat top? Let's be consistent.
    # Standard: vertices at angles 0,60,...,300 (pointy right). Face normals at 30,90,...,330.
    base_normals = np.deg2rad(30 + np.arange(0, 360, 60))
    normals = base_normals + r
    n_cos = np.cos(normals)
    n_sin = np.sin(normals)
    # For each point, compute max over normals of p·n
    # points: (M,2), normals: (6,)
    dots = points @ np.column_stack([n_cos, n_sin]).T  # (M, 6)
    max_dot = np.max(dots, axis=1)  # (M,)
    a = np.max(max_dot)  # required apothem
    s = 2.0 * a / np.sqrt(3.0)
    return s

def _total_points(inner_centers, inner_angles, side=1.0):
    """All vertices of all inner hexagons."""
    all_pts = []
    for cx, cy, ang in inner_centers:
        verts = _hex_vertices(cx, cy, side, ang)
        all_pts.append(verts)
    return np.vstack(all_pts)

def construct():
    # 3x4 honeycomb lattice positions.
    # Hex lattice basis: b1 = (sqrt(3), 0) * s, b2 = (sqrt(3)/2, 3/2) * s for side s=1.
    # For unit hexagons (side=1), center-to-center distance in honeycomb = sqrt(3).
    # Let's place them in a grid: rows 0..2, cols 0..3.
    # In a honeycomb, adjacent centers are sqrt(3) apart.
    # Use basis vectors: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 3/2)
    sqrt3 = np.sqrt(3.0)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3 / 2.0, 1.5])
    
    # 3 rows x 4 cols
    rows, cols = 3, 4
    base_positions = []
    for i in range(rows):
        for j in range(cols):
            pos = i * e1 + j * e2
            base_positions.append(pos)
    base_positions = np.array(base_positions)  # (12, 2)
    
    # Center the cluster
    centroid = np.mean(base_positions, axis=0)
    base_positions = base_positions - centroid
    
    # Parameters: shear (applied to lattice basis), rotation of whole cluster, 
    # and we can also rotate each hexagon (but for honeycomb, all same rotation is optimal).
    # Let's parameterize: 
    #   - global rotation theta (degrees) of the entire cluster
    #   - shear: we deform the lattice by mixing e1 and e2.
    # Actually, simpler: apply an affine transform to the base positions, then rotate.
    # Affine transform: A @ pos, where A is a 2x2 matrix close to identity.
    # To keep it low-dim: A = [[1, s], [0, 1]] (shear in x by s*y) or similar.
    # But shear might break the hexagonal symmetry. Let's use a general 2x2 but constrained.
    # Alternative: just optimize rotation of the whole cluster and the rotation of the outer hexagon.
    # The cluster shape is fixed (honeycomb 3x4). We rotate the cluster, and find best outer hex rotation.
    
    def objective(params):
        cluster_rot = params[0]  # degrees
        # Apply rotation to base positions
        r = np.radians(cluster_rot)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        positions = base_positions @ R.T
        
        # All vertices of inner hexagons (each hexagon rotated by cluster_rot too)
        all_pts = _total_points(
            [(float(p[0]), float(p[1]), float(cluster_rot)) for p in positions]
        )
        
        # Center the points (the cluster centroid should be at origin, but let's be safe)
        # Actually, we want the outer hexagon centered at origin. The cluster is already centered.
        # But the vertices might not be symmetric. Let's just use the points as-is.
        
        # Find best outer hexagon rotation to minimize enclosing side
        # This is a 1D optimization. Use a fine grid first, then refine.
        best_side = np.inf
        best_outer_rot = 0.0
        for outer_rot in np.linspace(0, 60, 121):
            s = _enclosing_side(all_pts, outer_rot)
            if s < best_side:
                best_side = s
                best_outer_rot = outer_rot
        
        # Refine
        if best_side < np.inf:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(
                lambda x: _enclosing_side(all_pts, x),
                bounds=(best_outer_rot - 3, best_outer_rot + 3),
                method='bounded'
            )
            if res.success:
                best_side = res.fun
                best_outer_rot = res.x
        
        return best_side
    
    # Optimize cluster rotation
    best_cluster_rot = 0.0
    best_val = np.inf
    for init_rot in np.linspace(0, 60, 13):
        res = minimize_scalar(
            objective,
            bounds=(init_rot - 3, init_rot + 3),
            method='bounded'
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_cluster_rot = res.x
    
    # Final refinement
    res = minimize_scalar(
        objective,
        bounds=(best_cluster_rot - 1, best_cluster_rot + 1),
        method='bounded'
    )
    if res.success:
        best_cluster_rot = res.x
        best_val = res.fun
    
    # Build final configuration
    r = np.radians(best_cluster_rot)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    positions = base_positions @ R.T
    
    inner = [[float(p[0]), float(p[1]), float(best_cluster_rot)] for p in positions]
    
    # Find best outer rotation
    all_pts = _total_points(
        [(float(p[0]), float(p[1]), float(best_cluster_rot)) for p in positions]
    )
    best_outer_rot = 0.0
    best_side = np.inf
    for outer_rot in np.linspace(0, 60, 121):
        s = _enclosing_side(all_pts, outer_rot)
        if s < best_side:
            best_side = s
            best_outer_rot = outer_rot
    res = minimize_scalar(
        lambda x: _enclosing_side(all_pts, x),
        bounds=(best_outer_rot - 3, best_outer_rot + 3),
        method='bounded'
    )
    if res.success:
        best_side = res.fun
        best_outer_rot = res.x
    
    return inner, (0.0, 0.0), float(best_side), float(best_outer_rot)
