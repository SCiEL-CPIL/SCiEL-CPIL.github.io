"""Packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    angles = angle + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def min_distance_between_hexes(hex1, hex2):
    """Approximate minimum distance between two hexagons using vertex-edge distances."""
    min_dist = np.inf
    for i in range(6):
        v1 = hex1[i]
        for j in range(6):
            v2 = hex2[j]
            d = np.linalg.norm(v1 - v2)
            if d < min_dist:
                min_dist = d
    return min_dist

def compute_outer_hex_side(inner_hexes, outer_center, outer_angle):
    """Compute minimum side length of outer hexagon containing all inner hexagons."""
    cx, cy = outer_center
    a = np.radians(outer_angle)
    normals = []
    for k in range(6):
        na = a + np.pi/6 + k * np.pi/3
        normals.append(np.array([np.cos(na), np.sin(na)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for hx, hy, ha in inner_hexes:
        verts = hex_vertices(hx, hy, 1.0, ha)
        verts_rel = verts - np.array(outer_center)
        projs = verts_rel @ normals.T
        max_projs = projs.max(axis=1)
        max_proj = max(max_proj, max_projs.max())
    
    side = max_proj / (np.sqrt(3) / 2)
    return side

def objective(params):
    """Objective: outer hexagon side length + penalty for overlaps."""
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2]
        inner.append((x, y, angle))
    
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_angle = params[3*N+2]
    
    side = compute_outer_hex_side(inner, (outer_cx, outer_cy), outer_angle)
    
    penalty = 0.0
    hex_verts = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance_between_hexes(hex_verts[i], hex_verts[j])
            if d < 0.1:
                penalty += (0.1 - d) * 50.0
    
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Honeycomb arrangement
    dx = 3 * np.sqrt(3) / 2
    dy = 1.5
    
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            positions.append((x, y, 0.0))
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    
    inner = [(x - cx0, y - cy0, a) for x, y, a in positions]
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    initial_side = compute_outer_hex_side(inner, outer_center, outer_angle)
    
    params0 = []
    for x, y, a in inner:
        params0.extend([x, y, a])
    params0.extend([outer_center[0], outer_center[1], outer_angle])
    params0 = np.array(params0)
    
    best_params = params0.copy()
    best_score = objective(params0)
    
    try:
        result = minimize(objective, params0, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() < deadline:
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
    except Exception:
        pass
    
    # Try different outer angles
    for trial_angle in [30, 60, 90, 120, 150]:
        if time.time() > deadline - 15:
            break
        params_test = best_params.copy()
        params_test[3*N+2] = float(trial_angle)
        try:
            result = minimize(objective, params_test, method='L-BFGS-B',
                             options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
            if time.time() < deadline:
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
        except Exception:
            pass
    
    inner_result = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner_result.append([x, y, angle])
    
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    
    side = compute_outer_hex_side(inner_result, (outer_cx, outer_cy), outer_angle)
    
    return inner_result, (outer_cx, outer_cy), side, outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
