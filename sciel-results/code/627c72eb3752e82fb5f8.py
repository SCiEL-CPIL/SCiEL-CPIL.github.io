import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 110.0  # safety margin

def hex_vertex_angles():
    """Return angles of the 6 vertices of a unit regular hexagon (flat-top)."""
    return np.array([np.pi/3 * i for i in range(6)])

def hex_vertices(angle_deg):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg."""
    a = np.radians(angle_deg)
    base = np.array([[np.cos(np.pi/3 * i), np.sin(np.pi/3 * i)] for i in range(6)])
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return base @ R.T

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    # Rotate point back
    rx = c * dx + s * dy
    ry = -s * dx + c * dy
    # Unit hexagon check (flat-top)
    if abs(ry) > side * np.sqrt(3)/2:
        return False
    if abs(rx) > side:
        return False
    if abs(rx) > side/2:
        # Check the slanted sides
        bound = side * np.sqrt(3) * (side - abs(rx)) / side
        if abs(ry) > side * np.sqrt(3)/2 - (np.sqrt(3)/2) * (abs(rx) - side/2) * 2 / 1:
            pass
    # More robust: use the 6 half-plane inequalities
    # For flat-top hex with side s, center at origin:
    # |x| <= s
    # |y| <= s*sqrt(3)/2
    # |x| + |y|/sqrt(3) <= s  (for the 4 corner regions)
    # Actually let's use: for each of 6 edges, check inside
    verts = np.array([[np.cos(np.pi/3*i), np.sin(np.pi/3*i)] for i in range(6)]) * side
    # For each edge, check the point is on the correct side
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i+1) % 6]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Check that point is on the same side as center (origin)
        dot_p = np.dot([rx, ry], normal)
        dot_v = np.dot(v1, normal)
        if dot_p > dot_v + 1e-12:
            return False
    return True

def min_outer_side(inner, oc, oa):
    """Compute the minimum side length of outer hexagon (centered at oc, angle oa) containing all inner hexagons."""
    # For each inner hexagon, we need to find the maximum "radius" in the direction of each face normal
    # The outer hexagon with side S, center C, angle A contains point P if for all 6 face normals n_i:
    #   dot(P - C, n_i) <= S * sqrt(3) / 2 * ... 
    # Actually, for a regular hexagon with side S, the apothem (inradius) is S * sqrt(3)/2.
    # The constraint is: for each of the 6 outward normals n_i, dot(P - C, n_i) <= S * sqrt(3)/2
    # So S >= max_i max_P dot(P - C, n_i) / (sqrt(3)/2)
    
    a = np.radians(oa)
    # Face normals (outward) for flat-top hex
    normals = np.array([[np.cos(np.pi/3 * i + np.pi/6), np.sin(np.pi/3 * i + np.pi/6)] for i in range(6)])
    normals = (normals @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T).T
    
    max_proj = 0.0
    for (px, py, ang) in inner:
        # Get vertices of this unit hexagon
        verts = hex_vertices(ang) + np.array([px, py])
        for v in verts:
            for n in normals:
                proj = np.dot(v - np.array(oc), n)
                if proj > max_proj:
                    max_proj = proj
    
    return max_proj / (np.sqrt(3) / 2)

def generate_axial_config(k):
    """Generate an axial-coordinate based configuration of N hexagons."""
    if k == 0:
        # Linear row
        return [[2.0 * i, 0.0, 0.0] for i in range(N)]
    elif k == 1:
        # Staggered two rows
        inner = []
        for i in range(N):
            row = i // 6
            col = i % 6
            x = 2.0 * col + (row % 2) * 1.0
            y = row * np.sqrt(3)
            inner.append([x, y, 0.0])
        return inner
    elif k == 2:
        # Compact cluster (honeycomb pattern)
        inner = []
        # Place in a compact hexagonal arrangement
        positions = []
        # Center + 6 around + 12 around = 19, but we only need 11
        # Use axial coordinates
        hex_positions = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= 3:
                    hex_positions.append((q, r))
        # Sort by distance from center
        hex_positions.sort(key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
        for q, r in hex_positions[:N]:
            x = 2.0 * q + r
            y = np.sqrt(3) * r
            inner.append([x, y, 0.0])
        return inner
    elif k == 3:
        # Linear with slight rotation
        angle = 15.0
        inner = []
        for i in range(N):
            t = i - N//2
            x = 2.0 * t * np.cos(np.radians(angle))
            y = 2.0 * t * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        return inner
    elif k == 4:
        # Two staggered rows
        inner = []
        for i in range(6):
            inner.append([2.0 * i, 0.0, 0.0])
        for i in range(5):
            inner.append([2.0 * i + 1.0, np.sqrt(3), 0.0])
        return inner
    else:
        # Variations
        offset = 0.5 * (k % 3)
        inner = [[(2.0 + offset) * (i - N//2), 0.0, 0.0] for i in range(N)]
        return inner

def params_to_inner(params):
    """Extract inner hexagon positions from parameter vector."""
    n = N
    inner = []
    for i in range(n):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner.append([x, y, ang])
    return inner

def objective(params):
    """Objective: minimize outer hexagon side length."""
    inner = params_to_inner(params)
    oc = (0.0, 0.0)  # Fix outer center at origin for symmetry
    oa = 0.0  # Fix outer angle at 0 for symmetry
    try:
        side = min_outer_side(inner, oc, oa)
    except Exception:
        return 1e10
    if not np.isfinite(side):
        return 1e10
    return side

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_params = None
    
    # Generate multiple initial configurations
    configs = []
    for k in range(8):
        configs.append(generate_axial_config(k))
    
    # Also add random perturbations of the best config
    for cfg in configs[:3]:
        for _ in range(3):
            perturbed = []
            for (x, y, a) in cfg:
                perturbed.append([
                    x + np.random.uniform(-0.5, 0.5),
                    y + np.random.uniform(-0.5, 0.5),
                    a + np.random.uniform(-10, 10)
                ])
            configs.append(perturbed)
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Convert to params (fix outer center and angle)
        params = np.zeros(3 * N)
        for i, (x, y, a) in enumerate(cfg):
            params[3*i] = x
            params[3*i+1] = y
            params[3*i+2] = a
        
        score = objective(params)
        if score < best_score:
            best_score = score
            best_params = params.copy()
        
        # Nelder-Mead optimization
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 20:
            try:
                result = minimize(
                    objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': int(min(800, remaining * 40)), 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
    
    # Final polish
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    inner = params_to_inner(best_params)
    return inner, (0.0, 0.0), best_score, 0.0
