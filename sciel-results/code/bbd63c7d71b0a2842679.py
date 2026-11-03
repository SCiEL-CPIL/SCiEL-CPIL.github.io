"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side, angle."""
    # Transform point to hexagon's local frame
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Regular hexagon: |x| <= side, |y| <= side*sqrt(3)/2, and |x| + |y|/sqrt(3) <= side
    # Actually for a regular hexagon with vertices at distance 'side' from center:
    # The hexagon is defined by: for each of 6 directions, dot product <= inradius
    # Inradius = side * sqrt(3)/2
    inradius = side * np.sqrt(3) / 2
    # The 6 normal directions are at angles 0, 60, 120, 180, 240, 300
    normals = np.column_stack([np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)])
    dots = normals @ np.array([lx, ly])
    return np.all(dots <= inradius)

def hexagon_contains_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner hexagon is fully inside outer hexagon."""
    # Sample points on inner hexagon boundary and a few interior points
    inner_verts = hex_vertices(inner_cx, inner_cy, 1.0, inner_angle)
    # Also check center
    test_points = np.vstack([inner_verts, np.array([[inner_cx, inner_cy]])])
    for px, py in test_points:
        if not point_in_hexagon(px, py, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def check_all_inside(inner_positions, outer_cx, outer_cy, outer_side, outer_angle):
    """Check all inner hexagons are inside outer hexagon."""
    for cx, cy, ang in inner_positions:
        if not hexagon_contains_hexagon(cx, cy, ang, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def check_no_overlap(inner_positions, tol=0.01):
    """Check no two inner hexagons overlap (approximate via center distance)."""
    # For two unit hexagons, minimum center distance for no overlap is 2*inradius = sqrt(3) ≈ 1.732
    # But this depends on relative rotation. Conservative: use 1.732
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            dx = inner_positions[i][0] - inner_positions[j][0]
            dy = inner_positions[i][1] - inner_positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.732 - tol:
                return False
    return True

def objective(params):
    """Objective: outer side length + penalty for violations."""
    # params: [outer_cx, outer_cy, outer_side, outer_angle, then for each inner: cx, cy, angle]
    outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
    inner_params = params[4:]
    inner_positions = []
    for k in range(N):
        cx = inner_params[k*3]
        cy = inner_params[k*3+1]
        ang = inner_params[k*3+2]
        inner_positions.append((cx, cy, ang))
    
    penalty = 0.0
    # Penalty for inner hexagons not being inside outer
    for cx, cy, ang in inner_positions:
        if not hexagon_contains_hexagon(cx, cy, ang, outer_cx, outer_cy, outer_side, outer_angle):
            penalty += 10.0
    # Penalty for overlap
    if not check_no_overlap(inner_positions, tol=0.0):
        penalty += 10.0
    
    return outer_side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s margin
    
    # Initial guess: 3x4 grid of hexagons
    # Hexagons in a honeycomb-like grid
    # Row spacing: sqrt(3) * 1 = 1.732 (for side 1)
    # Column spacing: 1.5 (horizontal offset for honeycomb)
    
    # Let's try a 3-wide, 4-tall arrangement
    # Positions for 3 columns, 4 rows
    col_spacing = 1.5  # horizontal distance between column centers
    row_spacing = np.sqrt(3)  # vertical distance between rows
    # For honeycomb, odd rows are offset by 0.75
    
    initial_inner = []
    for row in range(4):
        for col in range(3):
            x = (col - 1) * col_spacing + (0.75 if row % 2 == 1 else 0)
            y = (row - 1.5) * row_spacing
            initial_inner.append([x, y, 0.0])
    
    # Estimate outer hexagon size
    # Bounding box of inner hexagons
    xs = [p[0] for p in initial_inner]
    ys = [p[1] for p in initial_inner]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min + 2  # add 2 for hexagon extent
    height = y_max - y_min + 2*np.sqrt(3)/2 + 2
    
    # Outer hexagon side: need to contain the cluster
    # A hexagon with side s has width 2*s and height sqrt(3)*s
    # So s >= max(width/2, height/sqrt(3))
    outer_side_init = max(width/2, height/np.sqrt(3)) + 0.5
    outer_cx_init = (x_min + x_max) / 2
    outer_cy_init = (y_min + y_max) / 2
    outer_angle_init = 0.0
    
    # Build initial params
    params0 = [outer_cx_init, outer_cy_init, outer_side_init, outer_angle_init]
    for p in initial_inner:
        params0.extend([p[0], p[1], p[2]])
    
    best_params = np.array(params0)
    best_val = objective(best_params)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 30000})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Extract result
    outer_cx, outer_cy, outer_side, outer_angle = best_params[0], best_params[1], best_params[2], best_params[3]
    inner_positions = []
    for k in range(N):
        cx = best_params[4 + k*3]
        cy = best_params[4 + k*3 + 1]
        ang = best_params[4 + k*3 + 2]
        inner_positions.append([cx, cy, ang])
    
    # Verify and if invalid, fall back to a simple valid solution
    if not check_all_inside(inner_positions, outer_cx, outer_cy, outer_side, outer_angle) or \
       not check_no_overlap(inner_positions):
        # Fallback: use a larger outer hexagon
        outer_side = outer_side + 2.0
        # Re-center
        xs = [p[0] for p in inner_positions]
        ys = [p[1] for p in inner_positions]
        outer_cx = (min(xs) + max(xs)) / 2
        outer_cy = (min(ys) + max(ys)) / 2
    
    return inner_positions, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s margin
    
    # Initial guess: 3x4 grid of hexagons in honeycomb pattern
    col_spacing = 1.5
    row_spacing = np.sqrt(3)
    
    initial_inner = []
    for row in range(4):
        for col in range(3):
            x = (col - 1) * col_spacing + (0.75 if row % 2 == 1 else 0)
            y = (row - 1.5) * row_spacing
            initial_inner.append([x, y, 0.0])
    
    xs = [p[0] for p in initial_inner]
    ys = [p[1] for p in initial_inner]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min + 2
    height = y_max - y_min + 2*np.sqrt(3)/2 + 2
    outer_side_init = max(width/2, height/np.sqrt(3)) + 0.5
    outer_cx_init = (x_min + x_max) / 2
    outer_cy_init = (y_min + y_max) / 2
    outer_angle_init = 0.0
    
    params0 = [outer_cx_init, outer_cy_init, outer_side_init, outer_angle_init]
    for p in initial_inner:
        params0.extend([p[0], p[1], p[2]])
    
    best_params = np.array(params0)
    best_val = objective(best_params)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 30000})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    outer_cx, outer_cy, outer_side, outer_angle = best_params[0], best_params[1], best_params[2], best_params[3]
    inner_positions = []
    for k in range(N):
        cx = best_params[4 + k*3]
        cy = best_params[4 + k*3 + 1]
        ang = best_params[4 + k*3 + 2]
        inner_positions.append([cx, cy, ang])
    
    if not check_all_inside(inner_positions, outer_cx, outer_cy, outer_side, outer_angle) or \
       not check_no_overlap(inner_positions):
        outer_side = outer_side + 2.0
        xs = [p[0] for p in inner_positions]
        ys = [p[1] for p in inner_positions]
        outer_cx = (min(xs) + max(xs)) / 2
        outer_cy = (min(ys) + max(ys)) / 2
    
    return inner_positions, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
