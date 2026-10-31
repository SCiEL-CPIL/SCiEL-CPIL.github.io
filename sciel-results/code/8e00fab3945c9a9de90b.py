<<<<<<< SEARCH
import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def construct():
    # Generate 11 points on a triangular lattice with slight expansion to prevent overlap
    # Axial coordinates (q, r)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Scale factor slightly greater than 1 to ensure no overlap
    SCALE = 1.001
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
    
    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    
    # Function to compute minimal side length S for a given outer rotation beta (radians)
    def calc_S(beta):
        # Check time limit
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        # Normals of the outer hexagon
        # Outer hexagon is centered at centroid, rotated by beta
        # Normals are at beta + pi/6 + k*pi/3
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            # Project all points onto this normal
            # Relative to centroid
            rel_pts = pts - centroid
            projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = (abs_proj + APOTHEM) / APOTHEM
            if required_S > max_S:
                max_S = required_S
        
        return max_S

    # Minimize S over beta in [0, pi/6]
    try:
        res = minimize_scalar(calc_S, bounds=(0, math.pi / 6), method='bounded',
                             options={'maxiter': 100})
        best_beta = res.x
        best_S = res.fun
    except Exception:
        # Fallback: use beta = 0
        best_beta = 0.0
        best_S = calc_S(0.0)
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(centroid[0]), float(centroid[1])]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
=======
import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def hex_corners(cx, cy, angle_rad, side=1.0):
    """Get the 6 corners of a unit hexagon centered at (cx,cy) with rotation angle_rad."""
    angles = angle_rad + np.arange(6) * math.pi / 3.0
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def point_in_hexagon(px, py, cx, cy, side, angle_rad):
    """Check if point (px,py) is inside or on the boundary of a regular hexagon."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, the vertices are at distance s from center
    # The hexagon is the intersection of 6 half-planes
    # In local coords, the hexagon has vertices at (s*cos(k*pi/3), s*sin(k*pi/3))
    # The supporting lines are at distance s*sqrt(3)/2 from center, normal at angle k*pi/3 + pi/6
    
    for k in range(6):
        ang = k * math.pi / 3.0 + math.pi / 6.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        dist = lx * nx + ly * ny
        if dist > side * APOTHEM + 1e-10:
            return False
    return True

def check_disjointness(pts, angles, tol=1e-9):
    """Check that no two unit hexagons overlap. Returns True if disjoint."""
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        ai = angles[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            aj = angles[j]
            # Check if any corner of hex i is inside hex j, or vice versa
            corners_i_x, corners_i_y = hex_corners(xi, yi, ai)
            for k in range(6):
                if point_in_hexagon(corners_i_x[k], corners_i_y[k], xj, yj, 1.0, aj):
                    return False
            corners_j_x, corners_j_y = hex_corners(xj, yj, aj)
            for k in range(6):
                if point_in_hexagon(corners_j_x[k], corners_j_y[k], xi, yi, 1.0, ai):
                    return False
    return True

def construct():
    # Generate 11 points on a triangular lattice
    # Axial coordinates (q, r) for a compact cluster
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Initial positions on triangular lattice with exact spacing
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    
    pts = np.array(pts, dtype=float)
    angles = np.zeros(N)
    
    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    
    def compute_outer_side(pts, angles, center, outer_angle):
        """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
        # For each inner hexagon, find the maximum extent in each of the 6 normal directions
        max_extent = 0.0
        for k in range(6):
            ang = outer_angle + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            
            max_proj = -float('inf')
            min_proj = float('inf')
            
            for i in range(N):
                xi, yi = pts[i]
                ai = angles[i]
                # Corners of inner hexagon i
                cx_arr, cy_arr = hex_corners(xi, yi, ai)
                proj = cx_arr * nx + cy_arr * ny
                max_proj = max(max_proj, np.max(proj))
                min_proj = min(min_proj, np.min(proj))
            
            # The outer hexagon must contain [min_proj, max_proj] in this direction
            # The outer hexagon centered at 'center' with side S has apothem S*APOTHEM
            # In direction (nx,ny), the extent from center is S*APOTHEM
            # So we need: center_proj + S*APOTHEM >= max_proj
            #             center_proj - S*APOTHEM <= min_proj
            # => S*APOTHEM >= max(max_proj - center_proj, center_proj - min_proj)
            center_proj = center[0] * nx + center[1] * ny
            required = max(max_proj - center_proj, center_proj - min_proj) / APOTHEM
            max_extent = max(max_extent, required)
        
        return max_extent

    def objective(params):
        """Objective: minimize outer side length. params = [x0,y0,a0, x1,y1,a1, ..., cx, cy, outer_angle]"""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        n_params = 3 * N + 3
        if len(params) != n_params:
            return float('inf')
        
        # Extract positions and angles
        pos = params[:3*N].reshape(N, 3)
        cx_inner = pos[:, 0]
        cy_inner = pos[:, 1]
        ang_inner = pos[:, 2]
        
        # Outer center and angle
        outer_cx = params[3*N]
        outer_cy = params[3*N + 1]
        outer_angle = params[3*N + 2]
        
        # Check disjointness (this is expensive, so we use a penalty approach)
        # Actually, let's use a soft constraint: penalize overlaps heavily
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Check distance between centers; if too close, penalize
                dx = cx_inner[i] - cx_inner[j]
                dy = cy_inner[i] - cy_inner[j]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum center-to-center distance for non-overlapping unit hexagons
                # is 2*APOTHEM = sqrt(3) if aligned, but can be less if rotated
                # Use a conservative minimum of sqrt(3) * 0.95 to allow some rotation benefit
                min_dist = math.sqrt(3) * 0.98
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
        
        if overlap_penalty > 1e-6:
            return float('inf')
        
        center = np.array([outer_cx, outer_cy])
        return compute_outer_side(cx_inner, cy_inner, center, outer_angle)
    
    # Initial parameters
    init_params = []
    for i in range(N):
        init_params.extend([pts[i][0], pts[i][1], 0.0])
    init_params.extend([centroid[0], centroid[1], 0.0])
    init_params = np.array(init_params)
    
    # Try multiple starting points with different rotations
    best_result = None
    best_score = float('inf')
    
    # Start 1: aligned lattice
    starts = [
        init_params.copy(),
        # Start 2: slightly rotated inner hexagons
        np.array([init_params[i] if i % 3 != 2 else init_params[i] + math.pi/6 for i in range(len(init_params))]),
        # Start 3: different outer angle
        np.array([init_params[i] if i < 3*N + 2 else init_params[i] + math.pi/12 for i in range(len(init_params))]),
    ]
    
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score and res.fun < float('inf'):
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: use the initial lattice with optimal rotation
        # Just use the original approach
        def calc_S_simple(beta):
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            max_S = 0.0
            for nx, ny in normals:
                rel_pts = pts - centroid
                projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
                abs_proj = np.max(np.abs(projections))
                required_S = (abs_proj + APOTHEM) / APOTHEM
                if required_S > max_S:
                    max_S = required_S
            return max_S
        
        from scipy.optimize import minimize_scalar
        try:
            res = minimize_scalar(calc_S_simple, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100})
            best_beta = res.x
            best_S = res.fun
        except Exception:
            best_beta = 0.0
            best_S = calc_S_simple(0.0)
        
        inner = []
        angle_deg = math.degrees(best_beta)
        for x, y in pts:
            inner.append([float(x), float(y), angle_deg])
        
        center = [float(centroid[0]), float(centroid[1])]
        side = float(best_S)
        rotation = angle_deg
        
        return inner, center, side, rotation
    
    # Extract best result
    pos = best_result[:3*N].reshape(N, 3)
    outer_cx = best_result[3*N]
    outer_cy = best_result[3*N + 1]
    outer_angle = best_result[3*N + 2]
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), math.degrees(pos[i][2])])
    
    center = [float(outer_cx), float(outer_cy)]
    side = float(best_score)
    rotation = math.degrees(outer_angle)
    
    return inner, center, side, rotation
>>>>>>> REPLACE
