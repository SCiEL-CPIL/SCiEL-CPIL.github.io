import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices_flat_top(cx, cy, s, angle_rad=0.0):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), flat-top, rotated by angle_rad."""
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.radians([0, 60, 120, 180, 240, 300]) + angle_rad
    return np.array([[cx + s * np.cos(a), cy + s * np.sin(a)] for a in angles])

def hex_vertices_pointy_top(cx, cy, s, angle_rad=0.0):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), pointy-top, rotated by angle_rad."""
    # Pointy-top: vertices at angles 30, 90, 150, 210, 270, 330 degrees
    angles = np.radians([30, 90, 150, 210, 270, 330]) + angle_rad
    return np.array([[cx + s * np.cos(a), cy + s * np.sin(a)] for a in angles])

def polygon_area(vertices):
    """Compute area of a polygon given its vertices."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

def point_in_convex_polygon(point, vertices):
    """Check if point is inside a convex polygon using cross products."""
    n = len(vertices)
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        # Check if point is to the left of each edge (for CCW polygon)
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if cross < -1e-10:
            return False
    return True

def hexagon_contains_hexagon(inner_verts, outer_verts):
    """Check if inner hexagon is contained in outer hexagon (all vertices of inner are in outer)."""
    for v in inner_verts:
        if not point_in_convex_polygon(v, outer_verts):
            return False
    return True

def hexagons_disjoint(hex1_verts, hex2_verts, tolerance=1e-9):
    """Check if two hexagons are disjoint (no interior overlap)."""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_verts:
        if point_in_convex_polygon(v, hex2_verts):
            return False
    # Check if any vertex of hex2 is inside hex1
    for v in hex2_verts:
        if point_in_convex_polygon(v, hex1_verts):
            return False
    # Check for edge intersections
    n1 = len(hex1_verts)
    n2 = len(hex2_verts)
    for i in range(n1):
        a1 = hex1_verts[i]
        b1 = hex1_verts[(i + 1) % n1]
        for j in range(n2):
            a2 = hex2_verts[j]
            b2 = hex2_verts[(j + 1) % n2]
            if segments_intersect(a1, b1, a2, b2, tolerance):
                return False
    return True

def segments_intersect(p1, p2, p3, p4, tol=1e-9):
    """Check if line segments p1-p2 and p3-p4 intersect."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
       ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
        return True
    return False

def compute_outer_side_length(inner_positions, outer_center, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (centered at outer_center, 
    rotated by outer_angle) that contains all inner hexagons.
    inner_positions: list of (cx, cy, angle_deg) for each inner hexagon
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang_deg in inner_positions:
        ang_rad = np.radians(ang_deg)
        verts = hex_vertices_flat_top(cx, cy, 1.0, ang_rad)
        all_verts.append(verts)
    
    # The outer hexagon is defined by 6 half-planes. For a regular hexagon with side s,
    # center (ox, oy), rotation angle, the supporting lines are at distance s*sqrt(3)/2 
    # from center in 6 directions.
    # Actually, for a regular hexagon with side length s, the apothem (distance from center to side) is s*sqrt(3)/2.
    # The 6 normal directions are at angles: outer_angle, outer_angle+60, ..., outer_angle+300
    
    # For containment, each vertex of each inner hexagon must satisfy:
    # dot(v - outer_center, normal_i) <= apothem for all 6 normals i
    
    ox, oy = outer_center
    outer_angle_rad = np.radians(outer_angle)
    
    # The 6 outward normals of the outer hexagon
    normal_angles = [outer_angle_rad + np.radians(60 * k) for k in range(6)]
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    
    # For each vertex, compute the maximum required apothem
    max_apothem = 0.0
    for verts in all_verts:
        for v in verts:
            d = v - np.array([ox, oy])
            projections = normals @ d
            # The vertex must be within the hexagon, so projection along each normal <= apothem
            # Actually, for a convex hexagon, the constraint is that the projection onto each outward normal <= apothem
            # But we need to be careful with the sign
            # The outward normal points outward, so for containment: dot(v - c, n_i) <= apothem
            # But also we need dot(v - c, -n_i) <= apothem, which is equivalent to dot(v - c, n_i) >= -apothem
            # So the constraint is |dot(v - c, n_i)| <= apothem? No, that's not right for a hexagon.
            
            # Let me reconsider. A regular hexagon centered at c with apothem a and rotation angle theta
            # can be described as: for each of the 6 directions n_i (outward normals), 
            # dot(p - c, n_i) <= a
            # The 6 normals are at angles theta, theta+60, ..., theta+300
            
            # So for containment, we need: max over all vertices v and all i of dot(v - c, n_i) <= a
            # And a = s * sqrt(3) / 2, so s = 2 * a / sqrt(3)
            
            proj = projections
            max_proj = np.max(proj)
            if max_proj > max_apothem:
                max_apothem = max_proj
    
    # Also need to check the other direction? No, because the normals span all 6 sides.
    # Actually, I realize I need to be more careful. The hexagon is the intersection of 6 half-planes.
    # Each half-plane is defined by an outward normal n_i and the constraint dot(p-c, n_i) <= apothem.
    # So for all vertices v of inner hexagons: dot(v - c, n_i) <= apothem for all i.
    # The minimum apothem that satisfies this is max over all v, i of dot(v-c, n_i).
    
    if max_apothem < 0:
        max_apothem = 0.0
    
    side_length = 2.0 * max_apothem / np.sqrt(3.0)
    return side_length

def objective(params):
    """
    params: [s, ox, oy, outer_angle_deg, x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    where s is outer side length (not used in objective, just for reference)
    Actually, let's optimize positions and outer center/angle, and compute s from containment.
    
    Let me restructure: params = [ox, oy, outer_angle_deg, x1, y1, a1, ..., x12, y12, a12]
    Objective: compute the minimum outer side length that contains all inner hexagons.
    """
    ox = params[0]
    oy = params[1]
    outer_angle_deg = params[2]
    
    inner_positions = []
    for i in range(N):
        base = 3 + i * 3
        cx = params[base]
        cy = params[base + 1]
        ang = params[base + 2]
        inner_positions.append((cx, cy, ang))
    
    s = compute_outer_side_length(inner_positions, (ox, oy), outer_angle_deg)
    
    # Add penalty if any two inner hexagons overlap
    penalty = 0.0
    hex_verts = []
    for cx, cy, ang in inner_positions:
        ang_rad = np.radians(ang)
        verts = hex_vertices_flat_top(cx, cy, 1.0, ang_rad)
        hex_verts.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(hex_verts[i], hex_verts[j]):
                penalty += 1000.0
    
    return s + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial arrangement: hexagonal cluster
    # Use flat-top hexagons in a honeycomb pattern
    # Centers for a compact arrangement of 12 hexagons
    
    # Let's try a 3-4-3-2 arrangement or similar
    # For flat-top hexagons, horizontal spacing = 1.5, vertical = sqrt(3)
    # Odd columns offset by sqrt(3)/2
    
    sqrt3 = np.sqrt(3.0)
    
    # Try a hexagonal arrangement: 
    # Row 0: 2 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Row 4: 1 hexagon
    # Total: 12
    
    # Actually, let me use a more systematic approach
    # Place hexagons in a grid pattern with honeycomb offsets
    
    # Let's try: columns of heights 2, 3, 3, 2, 2 = 12
    # Or a more symmetric: 3, 3, 3, 3 = 12 in 4 columns
    
    # I'll use a simple initial guess and let optimization work
    
    # Initial positions: spread out in a hexagonal pattern
    initial_positions = []
    # Place 12 hexagons in a roughly hexagonal arrangement
    # Using axial coordinates for hex grid
    
    # Axial to pixel for flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    axial_coords = []
    # Generate 12 hexagons in a compact hexagonal shape
    # Hex numbers: 1, 7, 19... so 12 is between 7 and 19
    # Let's use a 3x4 rectangular-ish block
    
    # Positions in axial coords (q, r)
    # A compact shape for 12:
    coords = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        initial_positions.append([x, y, 0.0])
    
    # Build initial params
    params0 = [0.0, 0.0, 0.0]  # outer center and angle
    for pos in initial_positions:
        params0.extend([pos[0], pos[1], pos[2]])
    
    # Optimize
    best_params = params0
    best_score = objective(params0)
    
    # Use Nelder-Mead for optimization
    from scipy.optimize import minimize
    
    result = minimize(objective, params0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.fun < best_score:
        best_params = result.x
        best_score = result.fun
    
    # Try a few more restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Try different initial arrangements
        for trial in range(3):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb initial positions
            np.random.seed(trial)
            params_trial = params0.copy()
            for i in range(N):
                base = 3 + i * 3
                params_trial[base] += np.random.uniform(-0.5, 0.5)
                params_trial[base + 1] += np.random.uniform(-0.5, 0.5)
                params_trial[base + 2] += np.random.uniform(-30, 30)
            
            res = minimize(objective, params_trial, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
    
    # Extract best solution
    ox = best_params[0]
    oy = best_params[1]
    outer_angle = best_params[2]
    
    inner = []
    for i in range(N):
        base = 3 + i * 3
        cx = best_params[base]
        cy = best_params[base + 1]
        ang = best_params[base + 2]
        inner.append([cx, cy, ang])
    
    # Compute final side length
    side = compute_outer_side_length(inner, (ox, oy), outer_angle)
    
    return inner, (ox, oy), side, outer_angle
