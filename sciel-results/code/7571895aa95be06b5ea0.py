"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _support_hexagon(cx, cy, s, angle_deg, direction):
    """Support function of a regular hexagon in a given direction (unit vector)."""
    a = np.radians(angle_deg)
    # For a regular hexagon with side s, the support function in direction u is
    # s * max over vertices of (v - center) . u
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    vx = s * np.cos(angles)
    vy = s * np.sin(angles)
    proj = vx * direction[0] + vy * direction[1]
    return cx * direction[0] + cy * direction[1] + np.max(proj)

def _inner_dist_hex(cx1, cy1, a1, cx2, cy2, a2, direction):
    """
    Minimum distance along 'direction' between two unit hexagons.
    = support(hex1, dir) + support(hex2, -dir)
    If this is > 0, they are separated along dir.
    """
    a1r = np.radians(a1)
    a2r = np.radians(a2)
    angles1 = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a1r
    angles2 = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a2r
    
    # Support of hex1 in direction
    v1x = np.cos(angles1)
    v1y = np.sin(angles1)
    s1 = np.max(v1x * direction[0] + v1y * direction[1])
    
    # Support of hex2 in -direction
    v2x = np.cos(angles2)
    v2y = np.sin(angles2)
    s2 = np.max(v2x * (-direction[0]) + v2y * (-direction[1]))
    
    # Center separation projected on direction
    csep = (cx2 - cx1) * direction[0] + (cy2 - cy1) * direction[1]
    
    # They are separated if csep >= s1 + s2
    # The gap is csep - s1 - s2
    return csep - s1 - s2

def _check_valid(inner, center, side, angle_deg):
    """Check if all inner hexagons fit inside outer hexagon and don't overlap."""
    cx, cy = center
    outer_verts = _hex_vertices(cx, cy, side, angle_deg)
    
    # Check each inner hexagon is inside outer
    for i, (ix, iy, ia) in enumerate(inner):
        iv = _hex_vertices(ix, iy, 1.0, ia)
        # Check each vertex of inner is inside outer
        for j in range(6):
            px, py = iv[j]
            # Point in polygon test (outer is convex, so use support function approach)
            # For a convex polygon, point is inside if it's on the correct side of all edges
            inside = True
            for k in range(6):
                x1, y1 = outer_verts[k]
                x2, y2 = outer_verts[(k+1) % 6]
                # Edge normal (inward)
                ex, ey = x2 - x1, y2 - y1
                nx, ny = ey, -ex  # rotate 90 deg clockwise for inward normal if CCW
                # Actually for CCW polygon, inward normal is (ey, -ex) normalized... 
                # Let's use cross product
                cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check non-overlap between inner hexagons
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if _polygons_overlap(inner[i], inner[j]):
                return False
    return True

def _polygons_overlap(p1, p2):
    """Check if two convex polygons overlap using SAT (separating axis theorem)."""
    x1, y1, a1 = p1
    x2, y2, a2 = p2
    v1 = _hex_vertices(x1, y1, 1.0, a1)
    v2 = _hex_vertices(x2, y2, 1.0, a2)
    
    # Get all edge normals from both polygons
    edges = []
    for v in [v1, v2]:
        for i in range(6):
            x1, y1 = v[i]
            x2, y2 = v[(i+1) % 6]
            ex, ey = x2-x1, y2-y1
            length = np.sqrt(ex*ex + ey*ey)
            if length > 1e-12:
                edges.append((-ey/length, ex/length))
    
    for nx, ny in edges:
        # Project both polygons onto this axis
        proj1 = np.array([vx*nx + vy*ny for vx, vy in v1])
        proj2 = np.array([vx*nx + vy*ny for vx, vy in v2])
        
        if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
            return False  # Found separating axis
    
    return True

def _compute_min_side(inner, outer_center, outer_angle):
    """Compute minimum outer side length for given inner config and outer center/angle."""
    cx, cy = outer_center
    # For each inner hexagon, find the minimum side length such that it fits
    # The outer hexagon with center (cx,cy), side s, angle theta contains point p iff
    # for all 6 faces, the point is on the correct side.
    # Face k has outward normal n_k and offset d_k from center.
    # The constraint is: (p - center) . n_k <= d_k for all k
    # where d_k = s * cos(30°) = s * sqrt(3)/2 for a regular hexagon
    # Actually for a regular hexagon with side s, the distance from center to each face is s*sqrt(3)/2
    
    theta = np.radians(outer_angle)
    # Outward normals of the 6 faces of a regular hexagon
    # Faces are at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to vertices at 0,60,...)
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    
    min_s = 0.0
    for ix, iy, ia in inner:
        iv = _hex_vertices(ix, iy, 1.0, ia)
        for k in range(6):
            # For each vertex of inner hex, the constraint is:
            # (iv - center) . n_k <= s * sqrt(3)/2
            for vx, vy in iv:
                dot = (vx - cx) * normals[k, 0] + (vy - cy) * normals[k, 1]
                if dot > 0:
                    s_needed = dot / (np.sqrt(3) / 2)
                    min_s = max(min_s, s_needed)
    
    return min_s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    # Initial guess: hexagonal lattice arrangement
    # For 12 hexagons, try a 3x4 or 2x6 grid in hexagonal packing
    # Hexagonal lattice: spacing between centers is 2*sqrt(3) for non-overlapping unit hexagons
    # (distance between parallel faces of two adjacent unit hexagons = 2*sqrt(3))
    
    def make_hex_lattice(rows, cols, rotation_deg=0.0):
        """Create hexagonal lattice of unit hexagons."""
        spacing = 2 * np.sqrt(3)  # distance between centers of adjacent non-overlapping hexagons
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing
                y = r * spacing * 1.5
                if r % 2 == 1:
                    x += spacing / 2
                pts.append([x, y, rotation_deg])
        return pts
    
    # Try several initial configurations
    configs = []
    # 3x4 grid
    configs.append(make_hex_lattice(3, 4))
    # 4x3 grid  
    configs.append(make_hex_lattice(4, 3))
    # 2x6 grid
    configs.append(make_hex_lattice(2, 6))
    # 6x2 grid
    configs.append(make_hex_lattice(6, 2))
    
    # For each config, try to optimize
    for config_idx, config in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Center the config at origin
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        config_centered = [[p[0] - cx0, p[1] - cy0, p[2]] for p in config]
        
        # Initial outer side estimate
        s0 = _compute_min_side(config_centered, (0.0, 0.0), 0.0)
        
        # Use Nelder-Mead to optimize positions and rotations
        # Variables: for each inner hex: x, y, angle (3*N = 36 variables)
        # Plus outer center x, y and outer angle: 3 more = 39 total
        # But we can fix outer center at (0,0) by symmetry and optimize outer angle
        
        n_vars = 3 * N + 1  # 36 + 1 (outer angle)
        
        x0 = []
        for p in config_centered:
            x0.extend([p[0], p[1], np.radians(p[2])])
        x0.append(0.0)  # outer angle
        
        def objective(x):
            if time.time() > deadline:
                return 1e10
            inner_pts = []
            for i in range(N):
                ix = x[3*i]
                iy = x[3*i+1]
                ia = np.degrees(x[3*i+2])
                inner_pts.append([ix, iy, ia])
            outer_angle = np.degrees(x[3*N])
            
            # Check non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    if _polygons_overlap(inner_pts[i], inner_pts[j]):
                        return 1e10 + (i+j)
            
            s = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
            return s
        
        # Run Nelder-Mead
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            
            if result.fun < best_score and result.fun < 1e9:
                inner_pts = []
                for i in range(N):
                    ix = result.x[3*i]
                    iy = result.x[3*i+1]
                    ia = np.degrees(result.x[3*i+2])
                    inner_pts.append([ix, iy, ia])
                outer_angle = np.degrees(result.x[3*N])
                
                # Verify
                s_final = _compute_min_side(inner_pts, (0.0, 0.0), outer_angle)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _polygons_overlap(inner_pts[i], inner_pts[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and s_final < best_score:
                    best_score = s_final
                    best_inner = inner_pts
                    best_center = (0.0, 0.0)
                    best_side = s_final
                    best_angle = outer_angle
                    
        except Exception as e:
            pass
    
    # If we found something, return it
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback: return the seed
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
