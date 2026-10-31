import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0

def hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given rotation."""
    ang = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = ang + k * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    ang = math.radians(angle_deg)
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-ang)
    sin_a = math.sin(-ang)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Check against each edge of the hexagon
    for k in range(6):
        theta1 = k * math.pi / 3.0
        theta2 = (k + 1) * math.pi / 3.0
        # Edge from vertex k to vertex k+1
        x1 = side * math.cos(theta1)
        y1 = side * math.sin(theta1)
        x2 = side * math.cos(theta2)
        y2 = side * math.sin(theta2)
        
        # Edge normal (outward pointing)
        ex = x2 - x1
        ey = y2 - y1
        # Normal perpendicular to edge, pointing outward
        nx = -ey
        ny = ex
        # Normalize
        norm = math.sqrt(nx * nx + ny * ny)
        if norm < 1e-12:
            continue
        nx /= norm
        ny /= norm
        
        # Check if point is on the correct side of the edge
        # The edge midpoint
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        # Vector from edge midpoint to point
        vx = lx - mx
        vy = ly - my
        # Dot product with normal should be <= 0 for point inside
        if nx * vx + ny * vy > 1e-9:
            return False
    return True

def hexagons_overlap(v1, a1, r1, v2, a2, r2, tol=1e-7):
    """Check if two hexagons overlap using vertex containment tests."""
    # Check if any vertex of hex1 is inside hex2
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, v2[0], v2[1], r2, a2):
            return True
    # Check if any vertex of hex2 is inside hex1
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, v1[0], v1[1], r1, a1):
            return True
    return False

def compute_min_side_length(positions, angles, outer_center, outer_rotation_deg):
    """
    Compute the minimum side length S of the outer hexagon that contains all inner hexagons.
    positions: list of (x, y) for N inner hexagons
    angles: list of angle_degrees for N inner hexagons
    outer_center: (cx, cy) for outer hexagon
    outer_rotation_deg: rotation of outer hexagon
    Returns: minimum side length S
    """
    ocx, ocy = outer_center
    outer_ang = math.radians(outer_rotation_deg)
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(len(positions)):
        x, y = positions[i]
        verts = hexagon_vertices(x, y, angles[i], 1.0)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    # Outer hexagon normals: at angles outer_ang + pi/6 + k*pi/3
    max_S = 0.0
    for k in range(6):
        ang = outer_ang + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        
        # Project all vertices onto this normal, relative to outer center
        rel_x = all_verts[:, 0] - ocx
        rel_y = all_verts[:, 1] - ocy
        projections = rel_x * nx + rel_y * ny
        abs_proj = np.max(np.abs(projections))
        
        # Required side length: S * apothem >= abs_proj
        # So S >= abs_proj / apothem
        required_S = abs_proj / APOTHEM
        if required_S > max_S:
            max_S = required_S
    
    return max_S

def check_validity(positions, angles, outer_center, outer_rotation_deg, side_length):
    """Check if configuration is valid: no overlaps and all hexagons inside outer."""
    # Check overlaps
    for i in range(len(positions)):
        v1 = hexagon_vertices(positions[i][0], positions[i][1], angles[i], 1.0)
        for j in range(i + 1, len(positions)):
            v2 = hexagon_vertices(positions[j][0], positions[j][1], angles[j], 1.0)
            if hexagons_overlap(v1, positions[i][0], positions[i][1], v2, positions[j][0], positions[j][1], 1.0):
                return False
    
    # Check containment
    ocx, ocy = outer_center
    outer_ang = math.radians(outer_rotation_deg)
    for i in range(len(positions)):
        x, y = positions[i]
        verts = hexagon_vertices(x, y, angles[i], 1.0)
        for vx, vy in verts:
            # Check if vertex is inside outer hexagon
            dx = vx - ocx
            dy = vy - ocy
            cos_a = math.cos(-outer_ang)
            sin_a = math.sin(-outer_ang)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            inside = True
            for k in range(6):
                theta1 = k * math.pi / 3.0
                theta2 = (k + 1) * math.pi / 3.0
                x1 = side_length * math.cos(theta1)
                y1 = side_length * math.sin(theta1)
                x2 = side_length * math.cos(theta2)
                y2 = side_length * math.sin(theta2)
                
                ex = x2 - x1
                ey = y2 - y1
                nx = -ey
                ny = ex
                norm = math.sqrt(nx * nx + ny * ny)
                if norm < 1e-12:
                    continue
                nx /= norm
                ny /= norm
                
                mx = (x1 + x2) / 2.0
                my = (y1 + y2) / 2.0
                vx_local = lx - mx
                vy_local = ly - my
                if nx * vx_local + ny * vy_local > 1e-7:
                    inside = False
                    break
            
            if not inside:
                return False
    
    return True

def construct():
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initialize with triangular lattice
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    SCALE = 1.001
    init_pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        init_pts.append((x, y))
    
    init_pts = np.array(init_pts)
    centroid = np.mean(init_pts, axis=0)
    
    # Initial angles: all 0
    init_angles = [0.0] * N
    
    # Initial outer center: centroid
    init_outer_center = (float(centroid[0]), float(centroid[1]))
    init_outer_rotation = 0.0
    
    # Compute initial side length
    init_side = compute_min_side_length(
        [tuple(p) for p in init_pts], 
        init_angles, 
        init_outer_center, 
        init_outer_rotation
    )
    
    best_positions = [tuple(p) for p in init_pts]
    best_angles = init_angles.copy()
    best_outer_center = init_outer_center
    best_outer_rotation = init_outer_rotation
    best_side = init_side
    
    # Local search: optimize positions, angles, outer center, outer rotation
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Unpack parameters
        # 2*N positions, N angles, 2 outer center, 1 outer rotation
        pos_flat = params[:2*N]
        angles = params[2*N:3*N]
        outer_cx = params[3*N]
        outer_cy = params[3*N + 1]
        outer_rot = params[3*N + 2]
        
        positions = [(pos_flat[i*2], pos_flat[i*2+1]) for i in range(N)]
        
        # Check for overlaps (return large penalty if overlap)
        for i in range(N):
            v1 = hexagon_vertices(positions[i][0], positions[i][1], angles[i], 1.0)
            for j in range(i + 1, N):
                v2 = hexagon_vertices(positions[j][0], positions[j][1], angles[j], 1.0)
                if hexagons_overlap(v1, positions[i][0], positions[i][1], v2, positions[j][0], positions[j][1], 1.0):
                    return 1e6
        
        side = compute_min_side_length(positions, angles, (outer_cx, outer_cy), math.degrees(outer_rot))
        return side
    
    # Try local optimization
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5.0:
            # Start from initial configuration
            init_params = []
            for p in init_pts:
                init_params.extend([float(p[0]), float(p[1])])
            for a in init_angles:
                init_params.append(float(a))
            init_params.extend([float(centroid[0]), float(centroid[1]), 0.0])
            
            res = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if res.fun < best_side and res.fun < 1e5:
                # Extract results
                pos_flat = res.x[:2*N]
                new_angles = res.x[2*N:3*N]
                new_outer_cx = res.x[3*N]
                new_outer_cy = res.x[3*N + 1]
                new_outer_rot = res.x[3*N + 2]
                
                new_positions = [(pos_flat[i*2], pos_flat[i*2+1]) for i in range(N)]
                new_side = compute_min_side_length(new_positions, list(new_angles), (new_outer_cx, new_outer_cy), math.degrees(new_outer_rot))
                
                # Verify validity
                if check_validity(new_positions, list(new_angles), (new_outer_cx, new_outer_cy), math.degrees(new_outer_rot), new_side):
                    best_positions = new_positions
                    best_angles = list(new_angles)
                    best_outer_center = (float(new_outer_cx), float(new_outer_cy))
                    best_outer_rotation = float(new_outer_rot)
                    best_side = new_side
    
    except Exception:
        pass
    
    # Prepare output
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    center = [float(best_outer_center[0]), float(best_outer_center[1])]
    side = float(best_side)
    rotation = float(math.degrees(best_outer_rotation))
    
    return inner, center, side, rotation
