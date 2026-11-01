"""Packing 12 unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def point_in_hex(px, py, center, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon."""
    cx, cy = center
    dx, dy = px - cx, py - cy
    theta = np.radians(angle_deg)
    # Rotate point into hexagon's local frame
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    # Hexagon with flat-top orientation: vertices at angles 0,60,...
    # In local frame, hexagon has vertices at (s*cos(k*60), s*sin(k*60))
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with circumradius s, the inradius is s*sqrt(3)/2
    # The 6 faces are at distance s*sqrt(3)/2 from center, with normals at angles 30,90,...
    r_in = side * np.sqrt(3) / 2.0
    for k in range(6):
        a = np.radians(30 + 60 * k)
        nx, ny = np.cos(a), np.sin(a)
        if lx * nx + ly * ny > r_in:
            return False
    return True

def hexagon_overlap_depth(v1, v2):
    """
    Compute minimum overlap depth between two convex hexagons given their vertices.
    Returns 0 if separated, positive if overlapping.
    Uses separating axis theorem.
    """
    # Get all edge normals from both polygons
    normals = []
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
                normals.append(n)
    
    min_overlap = np.inf
    for n in normals:
        # Project both polygons onto normal
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return 0.0  # Separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return max(0.0, min_overlap)

def containment_violation(inner_verts, outer_center, outer_side, outer_angle):
    """
    Return max distance by which inner hexagon vertices violate containment.
    0 if all inside.
    """
    max_viol = 0.0
    for pt in inner_verts:
        if not point_in_hex(pt[0], pt[1], outer_center, outer_side, outer_angle):
            # Compute how far outside
            cx, cy = outer_center
            dx, dy = pt[0] - cx, pt[1] - cy
            theta = np.radians(outer_angle)
            cos_t, sin_t = np.cos(-theta), np.sin(-theta)
            lx = dx * cos_t - dy * sin_t
            ly = dx * sin_t + dy * cos_t
            r_in = outer_side * np.sqrt(3) / 2.0
            for k in range(6):
                a = np.radians(30 + 60 * k)
                nx, ny = np.cos(a), np.sin(a)
                dist = lx * nx + ly * ny - r_in
                if dist > max_viol:
                    max_viol = dist
    return max_viol

def objective(params):
    """
    params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, outer_x, outer_y, outer_side, outer_angle]
    Total: 12*3 + 4 = 40 parameters
    """
    if time.time() > DEADLINE:
        return 1e10
    
    inner_params = params[:36].reshape(12, 3)
    outer_x, outer_y, outer_side, outer_angle = params[36:40]
    
    if outer_side < 1.0:
        return 1e10
    
    outer_center = (outer_x, outer_y)
    
    total_penalty = 0.0
    
    # Compute inner hexagon vertices
    inner_verts_list = []
    for i in range(12):
        cx, cy, angle = inner_params[i]
        verts = hex_vertices((cx, cy), 1.0, angle)
        inner_verts_list.append(verts)
    
    # Containment violations
    for i in range(12):
        viol = containment_violation(inner_verts_list[i], outer_center, outer_side, outer_angle)
        if viol > 0:
            total_penalty += viol * viol
    
    # Overlap violations
    for i in range(12):
        for j in range(i + 1, 12):
            depth = hexagon_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                total_penalty += depth * depth
    
    # Objective: outer side + penalty
    return outer_side + 100.0 * total_penalty

def make_initial_config(style, outer_angle):
    """Generate initial configuration parameters."""
    params = np.zeros(40)
    
    if style == 'grid_3x4':
        # 3 rows x 4 columns
        dx = 2.0 * np.sqrt(3)  # spacing for flat-top hexagons
        dy = 1.5
        for row in range(3):
            for col in range(4):
                idx = row * 4 + col
                x = (col - 1.5) * dx
                y = (row - 1.0) * dy
                params[idx * 3] = x
                params[idx * 3 + 1] = y
                params[idx * 3 + 2] = 0.0
    
    elif style == 'hex_lattice':
        # Hexagonal close packing: rows of 4,4,4
        dx = 2.0 * np.sqrt(3)
        dy = 1.5
        for row in range(3):
            for col in range(4):
                idx = row * 4 + col
                x = (col - 1.5) * dx + (row % 2) * dx / 2
                y = (row - 1.0) * dy
                params[idx * 3] = x
                params[idx * 3 + 1] = y
                params[idx * 3 + 2] = 0.0
    
    elif style == 'ring':
        # 1 center + 6 around + 5 more
        # Actually 12: let's do 6 in a ring, 6 in outer ring
        for i in range(6):
            angle = i * 60.0
            r = 2.0 * np.sqrt(3)
            params[i * 3] = r * np.cos(np.radians(angle))
            params[i * 3 + 1] = r * np.sin(np.radians(angle))
            params[i * 3 + 2] = 0.0
        for i in range(6):
            angle = i * 60.0 + 30.0
            r = 4.0 * np.sqrt(3)
            params[(i + 6) * 3] = r * np.cos(np.radians(angle))
            params[(i + 6) * 3 + 1] = r * np.sin(np.radians(angle))
            params[(i + 6) * 3 + 2] = 0.0
    
    else:
        # Random scatter
        np.random.seed(42 + style)
        for i in range(12):
            params[i * 3] = np.random.uniform(-5, 5)
            params[i * 3 + 1] = np.random.uniform(-5, 5)
            params[i * 3 + 2] = np.random.uniform(0, 60)
    
    # Outer hexagon
    params[36] = 0.0  # outer_x
    params[37] = 0.0  # outer_y
    params[38] = 8.0  # outer_side (initial guess)
    params[39] = outer_angle
    
    return params

def construct():
    best_score = 1e10
    best_params = None
    
    styles = ['grid_3x4', 'hex_lattice', 'ring', 0, 1, 2]
    outer_angles = [0.0, 30.0]
    
    for style in styles:
        if time.time() > DEADLINE - 10:
            break
        
        for oa in outer_angles:
            if time.time() > DEADLINE - 5:
                break
            
            x0 = make_initial_config(style, oa)
            
            try:
                result = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
            except Exception:
                continue
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
    
    # If no good result, fall back to seed
    if best_params is None or best_score > 40.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract result
    inner_params = best_params[:36].reshape(12, 3)
    outer_x, outer_y, outer_side, outer_angle = best_params[36:40]
    
    # Final verification and cleanup
    inner = []
    for i in range(12):
        x, y, angle = inner_params[i]
        angle_deg = angle % 360.0
        if angle_deg > 180:
            angle_deg -= 360.0
        inner.append([float(x), float(y), float(angle_deg)])
    
    # Round to reasonable precision
    for i in range(12):
        inner[i][0] = round(inner[i][0], 10)
        inner[i][1] = round(inner[i][1], 10)
        inner[i][2] = round(inner[i][2], 10)
    
    outer_x = round(float(outer_x), 10)
    outer_y = round(float(outer_y), 10)
    outer_side = round(float(outer_side), 10)
    outer_angle = round(float(outer_angle), 10)
    
    return inner, (outer_x, outer_y), outer_side, outer_angle

