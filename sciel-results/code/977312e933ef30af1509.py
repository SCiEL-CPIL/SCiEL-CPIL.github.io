import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3
_SOFTMIN_BETA = 50.0

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with flat-top orientation offset by outer_angle_deg)
    that contains all points in verts. Uses LP to find optimal center and inradius.
    Returns (cx, cy, side) or None on failure.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = verts @ normals[k]
        
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _hex_distance(p1, a1, p2, a2):
    """Minimum distance between two unit hexagons at positions p1, p2 with angles a1, a2 (radians)."""
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

def _softmin(values, beta=50.0):
    """Soft-min approximation using log-sum-exp."""
    m = np.min(values)
    if m < -1000:
        return m
    shifted = values - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def _objective(params, outer_angle_deg):
    """Objective: outer hexagon side length for given inner positions and angles."""
    n = N
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        new_positions.append((x, y))
        new_angles.append(a)
    
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = _hex_distance(new_positions[i], new_angles[i], new_positions[j], new_angles[j])
            if dist < min_dist:
                min_dist = dist
    
    if min_dist < 0:
        return 1000.0 - min_dist * 100
    
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1], math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    if res:
        cx, cy, side = res
        return side
    return 1000.0

def _objective_smooth(params, outer_angle_deg, beta=50.0):
    """Smooth objective using soft-min for distance constraints."""
    n = N
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        new_positions.append((x, y))
        new_angles.append(a)
    
    all_dists = []
    for i in range(n):
        for j in range(i+1, n):
            dist = _hex_distance(new_positions[i], new_angles[i], new_positions[j], new_angles[j])
            all_dists.append(dist)
    
    if len(all_dists) == 0:
        all_dists = [1.0]
    
    sm = _softmin(np.array(all_dists), beta)
    
    if sm < 0:
        return 1000.0 - sm * 100
    
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1], math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    if res:
        cx, cy, side = res
        return side
    return 1000.0

def _evaluate_layout(positions, inner_angle, outer_angles_coarse, outer_angles_fine):
    """Evaluate a layout with given inner angle and search over outer angles."""
    all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search
    for outer_angle_deg in outer_angles_coarse:
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = outer_angle_deg
    
    # Fine search around best coarse angle
    if best_angle > 0 or best_side < float('inf'):
        fine_range = np.arange(max(0, best_angle - 2.0), min(30.1, best_angle + 2.0), 0.05)
        for outer_angle_deg in fine_range:
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = outer_angle_deg
    
    return best_side, best_center, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_layout_idx = -1
    best_inner_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Coarse outer angles
    coarse_angles = np.arange(0, 30.1, 1.0)
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
            
            local_side, local_center, local_angle = _evaluate_layout(positions, inner_angle, coarse_angles, [])
            
            if local_side < best_side:
                best_side = local_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = local_center
                best_outer_angle = local_angle
                best_layout_idx = layout_idx
                best_inner_angle = inner_angle
    
    # Local optimization to improve the best layout
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        positions = layouts[best_layout_idx]
        
        x0 = []
        for i in range(N):
            x0.append(positions[i][0])
            x0.append(positions[i][1])
            x0.append(math.radians(best_inner_angle))
        
        # Try a range of outer angles around the best
        outer_angles_to_try = list(np.arange(max(0, best_outer_angle - 10), min(30, best_outer_angle + 10), 2.0))
        
        for outer_angle_deg in outer_angles_to_try:
            if time.time() > _DEADLINE - 20.0:
                break
            
            try:
                # First pass: smooth objective
                res = minimize(_objective_smooth, x0, args=(outer_angle_deg, 50.0),
                              method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                # Second pass: exact objective
                if res.fun < best_side and res.fun < 100:
                    res2 = minimize(_objective, res.x, args=(outer_angle_deg,),
                                   method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res2.fun < best_side and res2.fun < 100:
                        new_positions = []
                        new_angles = []
                        for i in range(N):
                            new_positions.append((res2.x[3*i], res2.x[3*i+1]))
                            new_angles.append(math.degrees(res2.x[3*i+2]))
                        
                        all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                        res3 = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                        
                        if res3:
                            cx, cy, side = res3
                            if side < best_side:
                                best_side = side
                                best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                                best_center = (cx, cy)
                                best_outer_angle = outer_angle_deg
            except Exception:
                pass
    
    # Final high-precision polish pass
    if best_inner is not None and time.time() < _DEADLINE - 5.0:
        try:
            x0 = []
            for i in range(N):
                x0.append(best_inner[i][0])
                x0.append(best_inner[i][1])
                x0.append(math.radians(best_inner[i][2]))
            
            res = minimize(_objective, x0, args=(best_outer_angle,),
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_side and res.fun < 100:
                new_positions = []
                new_angles = []
                for i in range(N):
                    new_positions.append((res.x[3*i], res.x[3*i+1]))
                    new_angles.append(math.degrees(res.x[3*i+2]))
                
                all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                res2 = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                
                if res2:
                    cx, cy, side = res2
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
        except Exception:
            pass
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
