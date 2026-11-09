import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side length, and rotation (degrees)."""
    cx, cy = center
    a = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 plus rotation
    angles = a + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _max_inradius_projection(vertices, normal):
    """Max projection of vertices onto a unit normal direction."""
    return np.max(vertices @ normal)

def construct():
    # Generate 12 unit hexagons in a compact hexagonal lattice (pointy-top orientation)
    # Pointy-top hexagons: angle = 90 degrees (vertex pointing up)
    # Spacing: horizontal = sqrt(3), vertical = 1.5 for adjacent rows
    
    sqrt3 = np.sqrt(3.0)
    angle_inner = 90.0  # pointy-top
    
    positions = []
    # 3 rows: row 0 has 4, row 1 has 4, row 2 has 4
    # Row 0: y = 0, x = -1.5*sqrt3/2, -0.5*sqrt3/2, 0.5*sqrt3/2, 1.5*sqrt3/2
    # Row 1: y = 1.5, x = -sqrt3, 0, sqrt3, 2*sqrt3  -- wait, need proper offset
    
    # For pointy-top hexagons in a grid:
    # Column j, Row i:
    #   x = j * sqrt3 + (i % 2) * sqrt3/2
    #   y = i * 1.5
    
    # Let's do 4 columns x 3 rows
    for i in range(3):
        for j in range(4):
            x = j * sqrt3 + (i % 2) * (sqrt3 / 2.0)
            y = i * 1.5
            positions.append((x, y, angle_inner))
    
    # Center the cluster
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [(x - cx, y - cy, a) for (x, y, a) in positions]
    
    inner = [[p[0], p[1], p[2]] for p in positions]
    
    # Now find optimal enclosing hexagon
    # Outer hexagon params: center (ox, oy), side length s, rotation angle (degrees)
    # Constraint: all inner hexagon vertices must be inside the outer hexagon
    
    # For a regular hexagon with center c, side length s, rotation theta:
    # A point p is inside if for all 6 face normals n_k: (p - c) . n_k <= s * cos(0) = s
    # Actually, the inradius is s * sqrt(3)/2, so the constraint is (p-c).n_k <= s*sqrt(3)/2
    
    def _outer_face_normals(angle_deg):
        """Face normals of outer hexagon (perpendicular to each side)."""
        a = np.deg2rad(angle_deg)
        # Faces are at angles 30, 90, 150, 210, 270, 330 + rotation
        # Normal to each face points outward
        normals_angles = a + np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
        return normals
    
    def _check_containment(outer_params, inner_hexes):
        """Check if outer hexagon contains all inner hexagons. Return max violation (0 if contained)."""
        ox, oy, s, theta = outer_params
        if s < 0.5:
            return 1e10
        normals = _outer_face_normals(theta)
        inradius = s * np.sqrt(3.0) / 2.0
        max_violation = 0.0
        for (ix, iy, iangle) in inner_hexes:
            verts = _hex_vertices((ix, iy), 1.0, iangle)
            rel = verts - np.array([ox, oy])
            projections = rel @ normals.T  # shape (6, 6): 6 vertices x 6 normals
            # For each vertex, check against each face: projection <= inradius
            violations = projections - inradius
            max_violation = max(max_violation, np.max(violations))
        return max_violation
    
    # Initial guess
    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = _hex_vertices((ix, iy), 1.0, iangle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Estimate initial side length and rotation
    init_ox = 0.0
    init_oy = 0.0
    init_theta = 0.0
    
    # Find minimal s for given center and rotation
    def _min_s_for_params(ox, oy, theta):
        normals = _outer_face_normals(theta)
        rel = all_verts - np.array([ox, oy])
        projections = rel @ normals.T
        # Need s such that s*sqrt(3)/2 >= max projection for each normal
        required_inradii = np.max(projections, axis=0)  # per normal
        min_inradius = np.max(required_inradii)
        s = min_inradius * 2.0 / np.sqrt(3.0)
        return s
    
    init_s = _min_s_for_params(init_ox, init_oy, init_theta)
    
    # Optimize over (ox, oy, theta, s) to minimize s with constraint of containment
    def objective(params):
        ox, oy, theta, s = params
        if s < 0.1:
            return 1e10
        violation = _check_containment(params, inner)
        if violation > 0:
            return s + 1000 * violation
        return s
    
    # Try several initial rotations
    best_result = None
    for init_rot in [0, 30, 60, 90, 120, 150]:
        x0 = [0.0, 0.0, float(init_rot), _min_s_for_params(0.0, 0.0, float(init_rot))]
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if best_result is None or res.fun < best_result.fun:
            best_result = res
    
    ox, oy, theta, s = best_result.x
    # Ensure positive side
    if s < 0.5:
        s = _min_s_for_params(0.0, 0.0, 0.0)
    
    return inner, (float(ox), float(oy)), float(s), float(theta)

