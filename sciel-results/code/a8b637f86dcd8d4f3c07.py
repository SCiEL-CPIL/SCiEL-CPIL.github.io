"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    """Get 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_hexagon_local(px, py, s, angle_deg):
    """Check if point (px,py) in local coords is inside hex with side s, rotation angle_deg."""
    dx = px
    dy = py
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    apothem = s * np.sqrt(3) / 2
    normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    for na in normal_angles:
        nx = np.cos(na)
        ny = np.sin(na)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def hexagon_overlap_penalty(h1, h2, threshold=0.0):
    """Return penalty if two unit hexagons overlap. h = (x, y, angle_deg)."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    
    # Check if any vertex of h1 is inside h2, or vice versa
    # Also check edge-edge intersections for robustness
    # Simple check: vertices of one inside the other
    
    verts1 = hex_vertices(x1, y1, 1.0, a1)
    verts2 = hex_vertices(x2, y2, 1.0, a2)
    
    penalty = 0.0
    
    # Check vertices of hex1 inside hex2
    for vx, vy in verts1:
        if point_in_hexagon_local(vx - x2, vy - y2, 1.0, a2):
            penalty += 1.0
    
    # Check vertices of hex2 inside hex1
    for vx, vy in verts2:
        if point_in_hexagon_local(vx - x1, vy - y1, 1.0, a1):
            penalty += 1.0
    
    # Also check center-to-center distance as a quick rejection
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    # Minimum distance between centers for non-overlapping unit hexagons is sqrt(3)
    # But this depends on orientation. Use a conservative bound.
    if dist < np.sqrt(3) * 0.9:
        penalty += (np.sqrt(3) - dist) ** 2
    
    return penalty

def all_overlap_penalty(inner):
    """Total overlap penalty for all pairs of inner hexagons."""
    total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            total += hexagon_overlap_penalty(inner[i], inner[j])
    return total

def get_all_inner_vertices(inner):
    """Get all 6*12 vertices of all inner hexagons."""
    all_verts = []
    for (x, y, ang) in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    return np.vstack(all_verts)

def required_outer_side(inner, outer_cx, outer_cy, outer_angle):
    """Find minimum outer side length s such that all inner hex vertices fit inside outer hex."""
    all_verts = get_all_inner_vertices(inner)
    dx = all_verts[:, 0] - outer_cx
    dy = all_verts[:, 1] - outer_cy
    angle_rad = np.radians(outer_angle)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    max_val = 0.0
    for na in normal_angles:
        nx = np.cos(na)
        ny = np.sin(na)
        vals = lx * nx + ly * ny
        max_val = max(max_val, np.max(vals))
    
    return 2.0 / np.sqrt(3) * max_val

def objective(params):
    """Objective: minimize outer hex side length + penalty for overlaps.
    params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12, cx, cy, outer_angle]
    Total: 12*3 + 3 = 39 params
    """
    inner_params = params[:36].reshape(12, 3)
    outer_cx = params[36]
    outer_cy = params[37]
    outer_angle = params[38]
    
    inner = [list(inner_params[i]) for i in range(12)]
    
    s = required_outer_side(inner, outer_cx, outer_cy, outer_angle)
    overlap_pen = all_overlap_penalty(inner)
    
    # Heavy penalty for overlaps
    return s + 100.0 * overlap_pen

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Initial guess: hexagonal lattice arrangement
    init_params = np.zeros(39)
    col_spacing = np.sqrt(3)
    row_spacing = 1.5
    
    # 3 rows, 4 columns each, staggered
    for row in range(3):
        for col in range(4):
            idx = row * 4 + col
            x = (col - 1.5) * col_spacing
            y = (row - 1.0) * row_spacing
            if row % 2 == 1:
                x += col_spacing / 2
            init_params[idx * 3] = x
            init_params[idx * 3 + 1] = y
            init_params[idx * 3 + 2] = 0.0
    
    init_params[36] = 0.0
    init_params[37] = 0.0
    init_params[38] = 0.0
    
    best_params = init_params.copy()
    best_score = objective(best_params)
    
    # Try multiple optimization runs
    methods_to_try = ['L-BFGS-B', 'Nelder-Mead']
    n_restarts = 3
    
    for method in methods_to_try:
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            if restart == 0:
                x0 = init_params.copy()
            else:
                x0 = init_params + np.random.randn(39) * 0.3
            
            try:
                if method == 'L-BFGS-B':
                    result = minimize(objective, x0, method='L-BFGS-B', 
                                     options={'maxiter': 100, 'ftol': 1e-8})
                else:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
                    
            except Exception:
                pass
            
            if time.time() - start_time > DEADLINE:
                break
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Extract best solution
    inner_params = best_params[:36].reshape(12, 3)
    inner = []
    for i in range(12):
        x, y, ang = inner_params[i]
        inner.append([float(x), float(y), float(ang)])
    
    outer_cx = float(best_params[36])
    outer_cy = float(best_params[37])
    outer_angle = float(best_params[38])
    
    # Compute actual outer side (without penalty)
    outer_side = required_outer_side(inner, outer_cx, outer_cy, outer_angle)
    
    return inner, (outer_cx, outer_cy), float(outer_side), float(outer_angle)
