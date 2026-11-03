import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon (rotated by outer_angle_deg) containing all points."""
    # The 6 outward normals of the enclosing hexagon
    base_angle = math.radians(outer_angle_deg) + math.pi / 6
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            # -n_k · center - r <= -n_k · vi  =>  -n_k·c - r <= -n_k·vi
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            # n_k · center - r <= n_k · vi  =>  n_k·c - r <= n_k·vi
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _hex_distance(p1, a1, p2, a2):
    """Minimum distance between two unit hexagons (radians for angles)."""
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    
    def support(a, theta):
        vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
        return np.max(vals)
    
    h1 = support(a1, theta)
    h2 = support(a2, theta + math.pi)
    return d - h1 - h2

def _build_hexagonal_layout(scale=1.0):
    """1 center + 6 ring1 + 4 from ring2 (closest 4)."""
    s3 = _SQRT3 * scale
    u = (s3, 0.0)
    v = (s3/2, 1.5*scale)
    
    positions = [(0, 0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    for ti, tj in ring1:
        positions.append((ti*u[0]+tj*v[0], ti*u[1]+tj*v[1]))
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2+p[1]**2)
    for ti, tj in ring2_sorted[:4]:
        positions.append((ti*u[0]+tj*v[0], ti*u[1]+tj*v[1]))
    return positions[:N]

def _build_row_layout(scale=1.0):
    """Row-based layout: 3+5+3."""
    s3 = _SQRT3 * scale
    positions = []
    for k in range(-1, 2):
        positions.append((k*s3 + s3/2, -1.5*scale))
    for k in range(-2, 3):
        positions.append((k*s3, 0.0))
    for k in range(-1, 2):
        positions.append((k*s3 + s3/2, 1.5*scale))
    return positions[:N]

def _build_compact_layout(scale=1.0):
    """Try 2+3+3+3 or similar compact."""
    s3 = _SQRT3 * scale
    positions = []
    # Row 0: 3 hexagons
    for k in range(-1, 2):
        positions.append((k*s3, -1.5*scale))
    # Row 1: 4 hexagons (shifted)
    for k in range(-1, 3):
        positions.append((k*s3 + s3/2, 0.0))
    # Row 2: 4 hexagons
    for k in range(-1, 3):
        positions.append((k*s3, 1.5*scale))
    return positions[:N]

def _evaluate_config(positions, angles, outer_angle_deg):
    """Evaluate a configuration: return (side, center, valid)."""
    all_vertices = []
    for i in range(len(positions)):
        px, py = positions[i]
        a_deg = math.degrees(angles[i])
        verts = _hex_vertices(px, py, a_deg)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    
    # Check validity: no overlaps
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = _hex_distance(np.array(positions[i]), angles[i], np.array(positions[j]), angles[j])
            if d < min_dist:
                min_dist = d
    valid = min_dist >= -1e-9
    return side, (cx, cy), valid, min_dist

def _objective(params, outer_angle_deg):
    """Optimization objective: side length + penalty for overlaps."""
    n = N
    positions = []
    angles = []
    for i in range(n):
        positions.append((params[3*i], params[3*i+1]))
        angles.append(params[3*i+2])
    
    # Check overlaps
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = _hex_distance(np.array(positions[i]), angles[i], np.array(positions[j]), angles[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0:
        return 100.0 - min_dist * 50
    
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(positions[i][0], positions[i][1], math.degrees(angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side

def construct():
    if time.time() > _DEADLINE - 3.0:
        positions = _build_hexagonal_layout()
        inner = [[p[0], p[1], 0.0] for p in positions]
        all_verts = np.vstack([_hex_vertices(p[0], p[1], 0.0) for p in positions])
        cx, cy, side = _min_enclosing_hex_side_lp(all_verts, 0.0)
        return inner, (cx, cy), side, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    # Generate initial layouts
    layouts = [
        _build_hexagonal_layout(1.0),
        _build_row_layout(1.0),
        _build_compact_layout(1.0),
        _build_hexagonal_layout(1.001),
        _build_row_layout(1.001),
    ]
    
    # Phase 1: Evaluate initial layouts with various outer angles
    for positions in layouts:
        if time.time() > _DEADLINE - 30.0:
            break
        
        # Try inner angle 0 and 30
        for inner_angle_deg in [0.0, 30.0]:
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle_deg) for px, py in positions])
            
            for outer_angle_deg in np.arange(0, 30.0, 0.5):
                if time.time() > _DEADLINE - 30.0:
                    break
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                
                # Check validity
                angles = [math.radians(inner_angle_deg)] * N
                valid, min_dist = True, float('inf')
                for i in range(N):
                    for j in range(i+1, N):
                        d = _hex_distance(np.array(positions[i]), angles[i], np.array(positions[j]), angles[j])
                        if d < min_dist:
                            min_dist = d
                if min_dist < -1e-6:
                    continue
                
                if side < best_side:
                    best_side = side
                    best_inner = [[px, py, inner_angle_deg] for px, py in positions]
                    best_center = (cx, cy)
                    best_outer_angle = outer_angle_deg
                    best_params = []
                    for px, py in positions:
                        best_params.extend([px, py, math.radians(inner_angle_deg)])
    
    if best_inner is None:
        positions = _build_hexagonal_layout()
        inner = [[p[0], p[1], 0.0] for p in positions]
        all_verts = np.vstack([_hex_vertices(p[0], p[1], 0.0) for p in positions])
        cx, cy, side = _min_enclosing_hex_side_lp(all_verts, 0.0)
        return inner, (cx, cy), side, 0.0
    
    # Phase 2: Local optimization
    if time.time() < _DEADLINE - 20.0 and best_params is not None:
        x0 = best_params[:]
        
        for outer_angle_deg in [best_outer_angle]:
            if time.time() > _DEADLINE - 20.0:
                break
            
            try:
                res = minimize(_objective, x0, args=(outer_angle_deg,),
                              method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_side and res.fun < 50:
                    positions = [(res.x[3*i], res.x[3*i+1]) for i in range(N)]
                    angles = [res.x[3*i+2] for i in range(N)]
                    side, center, valid, min_dist = _evaluate_config(positions, angles, outer_angle_deg)
                    if valid and side < best_side:
                        best_side = side
                        best_inner = [[px, py, math.degrees(a)] for (px, py), a in zip(positions, angles)]
                        best_center = center
                        best_params = res.x[:]
            except Exception:
                pass
            
            # Nelder-Mead refinement
            if time.time() < _DEADLINE - 15.0:
                try:
                    nm_res = minimize(_objective, best_params, args=(outer_angle_deg,),
                                     method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                    if nm_res.fun < best_side and nm_res.fun < 50:
                        positions = [(nm_res.x[3*i], nm_res.x[3*i+1]) for i in range(N)]
                        angles = [nm_res.x[3*i+2] for i in range(N)]
                        side, center, valid, min_dist = _evaluate_config(positions, angles, outer_angle_deg)
                        if valid and side < best_side:
                            best_side = side
                            best_inner = [[px, py, math.degrees(a)] for (px, py), a in zip(positions, angles)]
                            best_center = center
                            best_params = nm_res.x[:]
                except Exception:
                    pass
    
    # Phase 3: Try different outer angles with current best positions
    if time.time() < _DEADLINE - 10.0 and best_params is not None:
        positions = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
        angles = [best_params[3*i+2] for i in range(N)]
        
        for outer_angle_deg in np.arange(0, 30.0, 0.25):
            if time.time() > _DEADLINE - 10.0:
                break
            side, center, valid, min_dist = _evaluate_config(positions, angles, outer_angle_deg)
            if valid and side < best_side:
                best_side = side
                best_inner = [[px, py, math.degrees(a)] for (px, py), a in zip(positions, angles)]
                best_center = center
                best_outer_angle = outer_angle_deg
    
    # Phase 4: Final polish - try small perturbations
    if time.time() < _DEADLINE - 5.0 and best_params is not None:
        for pert in [0.01, -0.01, 0.02, -0.02]:
            if time.time() > _DEADLINE - 5.0:
                break
            test_params = best_params[:]
            for i in range(N):
                test_params[3*i] += pert * 0.1
                test_params[3*i+1] += pert * 0.1
            
            positions = [(test_params[3*i], test_params[3*i+1]) for i in range(N)]
            angles = [test_params[3*i+2] for i in range(N)]
            side, center, valid, min_dist = _evaluate_config(positions, angles, best_outer_angle)
            if valid and side < best_side:
                best_side = side
                best_inner = [[px, py, math.degrees(a)] for (px, py), a in zip(positions, angles)]
                best_center = center
                best_params = test_params[:]
    
    # Final verification
    if best_inner is not None:
        positions = [(p[0], p[1]) for p in best_inner]
        angles = [math.radians(p[2]) for p in best_inner]
        side, center, valid, min_dist = _evaluate_config(positions, angles, best_outer_angle)
        if valid:
            best_side = side
            best_center = center
    
    return best_inner, best_center, best_side, best_outer_angle
