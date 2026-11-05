import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
EVALUATOR_TIMEOUT = 150.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _min_enclosing_hex_side(inner_params, outer_angle_deg, beta=0.0):
    n = len(inner_params)
    apothem_factor = SQRT3 / 2.0
    
    normal_angles = np.deg2rad(outer_angle_deg + 30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    hex_vertex_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    hex_vertex_cos = np.cos(hex_vertex_angles)
    hex_vertex_sin = np.sin(hex_vertex_angles)
    
    centers = inner_params[:, :2]
    rotations = inner_params[:, 2]
    
    cos_r = np.cos(np.deg2rad(rotations))
    sin_r = np.sin(np.deg2rad(rotations))
    
    local_vertices = np.column_stack([hex_vertex_cos, hex_vertex_sin])
    
    x_rot = cos_r[:, None] * local_vertices[None, :, 0] - sin_r[:, None] * local_vertices[None, :, 1]
    y_rot = sin_r[:, None] * local_vertices[None, :, 0] + cos_r[:, None] * local_vertices[None, :, 1]
    
    vx = centers[:, 0:1] + x_rot
    vy = centers[:, 1:2] + y_rot
    
    proj = normals[None, None, 0] * vx[:, :, None] + normals[None, None, 1] * vy[:, :, None]
    
    required_s = proj / apothem_factor
    required_s = np.where(required_s > 0, required_s, -np.inf)
    
    flat = required_s.flatten()
    
    if beta > 0:
        max_arr = np.max(flat)
        if max_arr == -np.inf or max_arr < 0:
            return 0.0
        shifted = beta * (flat - max_arr)
        shifted = np.clip(shifted, -100, 100)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_arr + log_sum / beta
    else:
        return np.max(flat)

def _compute_side(inner_params, outer_angle_deg, beta=0.0):
    return _min_enclosing_hex_side(inner_params, outer_angle_deg, beta)

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i+1, n):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """Concentric ring seed."""
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append(((SQRT3 + eps) * np.cos(angle), (SQRT3 + eps) * np.sin(angle)))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append(((3.0 + 2*eps) * np.cos(angle), (3.0 + 2*eps) * np.sin(angle)))
    
    inner = [[x, y, 30.0] for x, y in centers]
    
    side = _compute_side(np.array(inner), 30.0)
    return inner, (0.0, 0.0), side, 30.0

def _lattice_packing():
    """Hexagonal lattice seed with angle optimization."""
    v1 = np.array([SQRT3, 0.0])
    v2 = np.array([SQRT3/2, 1.5])
    
    selected = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (0,-2)]
    
    inner = []
    for (i, j) in selected:
        pos = i * v1 + j * v2
        inner.append([float(pos[0]), float(pos[1]), 30.0])
    
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 0.5):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    for angle_deg in np.arange(best_angle - 1, best_angle + 1, 0.05):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (0.0, 0.0), best_side, best_angle

def _compact_packing():
    """Try a more compact arrangement."""
    # Center + 6 around at distance SQRT3, then 4 more placed optimally
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((SQRT3 * np.cos(angle), SQRT3 * np.sin(angle)))
    # 4 more: place at 120-degree intervals at radius ~2.6
    for k in range(4):
        angle = np.deg2rad(30 + k * 90)
        centers.append((2.6 * np.cos(angle), 2.6 * np.sin(angle)))
    
    inner = [[x, y, 30.0] for x, y in centers]
    
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in np.arange(0, 60, 1.0):
        side = _compute_side(np.array(inner), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return inner, (0.0, 0.0), best_side, best_angle

def _overlap_penalty(inner_params, penalty_weight=50.0):
    n = len(inner_params)
    penalty = 0.0
    min_center_dist = SQRT3 - 0.005
    threshold = 0.02
    
    for i in range(n):
        x1, y1 = inner_params[i, 0], inner_params[i, 1]
        for j in range(i+1, n):
            x2, y2 = inner_params[j, 0], inner_params[j, 1]
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < min_center_dist + threshold:
                excess = min_center_dist + threshold - dist
                penalty += penalty_weight * excess * excess / threshold
    
    return penalty

def _objective(params, outer_angle_deg, n, beta=30.0):
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg, beta)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _coordinate_descent(inner_arr, outer_angle, time_limit, beta=50.0):
    """Coordinate descent on individual hexagon positions and rotations."""
    n = len(inner_arr)
    params = inner_arr.copy()
    current_side = _compute_side(params, outer_angle, 0.0)
    
    for step in range(3000):
        if time.time() > time_limit:
            break
        
        improved = False
        for i in range(n):
            # Optimize x
            best_x = params[i, 0]
            best_val = current_side
            for dx in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1, -0.2, 0.2]:
                params[i, 0] = best_x + dx
                val = _compute_side(params, outer_angle, beta) + _overlap_penalty(params)
                if val < best_val - 1e-12:
                    best_val = val
                    best_x = params[i, 0]
            params[i, 0] = best_x
            
            # Optimize y
            best_y = params[i, 1]
            best_val = _compute_side(params, outer_angle, 0.0) + _overlap_penalty(params)
            for dy in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1, -0.2, 0.2]:
                params[i, 1] = best_y + dy
                val = _compute_side(params, outer_angle, beta) + _overlap_penalty(params)
                if val < best_val - 1e-12:
                    best_val = val
                    best_y = params[i, 1]
            params[i, 1] = best_y
            
            # Optimize angle
            best_a = params[i, 2]
            best_val = _compute_side(params, outer_angle, 0.0) + _overlap_penalty(params)
            for da in [-5.0, -2.0, -0.5, 0.5, 2.0, 5.0, -10.0, 10.0]:
                params[i, 2] = (best_a + da) % 360
                val = _compute_side(params, outer_angle, beta) + _overlap_penalty(params)
                if val < best_val - 1e-12:
                    best_val = val
                    best_a = params[i, 2]
            params[i, 2] = best_a % 360
        
        new_side = _compute_side(params, outer_angle, 0.0)
        new_pen = _overlap_penalty(params)
        if new_side + new_pen >= current_side + 1e-10:
            break
        current_side = new_side
        improved = True
    
    return params, _compute_side(params, outer_angle, 0.0)

def _polish_angle(inner_arr, initial_angle, time_limit):
    best_side = _compute_side(inner_arr, initial_angle)
    best_angle = initial_angle
    
    for angle_deg in np.arange(initial_angle - 3, initial_angle + 3, 0.2):
        if time.time() > time_limit:
            break
        side = _compute_side(inner_arr, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    for angle_deg in np.arange(best_angle - 0.2, best_angle + 0.2, 0.02):
        if time.time() > time_limit:
            break
        side = _compute_side(inner_arr, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _compute_side(inner_arr, a),
                bounds=(best_angle - 0.05, best_angle + 0.05),
                method='bounded',
                options={'xatol': 1e-12}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_angle = res.x
        except Exception:
            pass
    
    return best_side, best_angle

def construct():
    start_time = time.time()
    DEADLINE = EVALUATOR_TIMEOUT - 5.0
    safety = 3.0
    
    n = 11
    
    # Generate multiple seed candidates
    seeds = []
    
    s1_inner, _, s1_side, s1_angle = _seed_solution()
    seeds.append((s1_inner, s1_side, s1_angle))
    
    s2_inner, _, s2_side, s2_angle = _lattice_packing()
    seeds.append((s2_inner, s2_side, s2_angle))
    
    s3_inner, _, s3_side, s3_angle = _compact_packing()
    seeds.append((s3_inner, s3_side, s3_angle))
    
    best_seed = min(seeds, key=lambda x: x[1])
    best_inner = np.array(best_seed[0], dtype=float)
    best_side = best_seed[1]
    best_angle = best_seed[2]
    
    # Phase 1: Nelder-Mead with soft-max surrogate
    if time.time() < DEADLINE - safety - 30:
        for outer_angle in [best_angle, 0.0, 15.0, 30.0, 45.0]:
            if time.time() > DEADLINE - safety - 15:
                break
            try:
                init = best_inner.flatten().copy()
                res = minimize(
                    lambda p: _objective(p, outer_angle, n, beta=40.0),
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                if time.time() > DEADLINE - safety:
                    break
                cand = res.x.reshape(n, 3)
                cand_side = _compute_side(cand, outer_angle, 0.0)
                if _is_valid(cand, cand_side, outer_angle, min_sep=1e-9):
                    if cand_side < best_side:
                        best_inner = cand
                        best_side = cand_side
                        best_angle = outer_angle
            except Exception:
                pass
    
    # Phase 2: Angle polish
    if time.time() < DEADLINE - safety:
        best_side, best_angle = _polish_angle(best_inner, best_angle, DEADLINE - safety)
    
    # Phase 3: Coordinate descent for fine-tuning
    if time.time() < DEADLINE - safety - 10:
        best_inner, new_side = _coordinate_descent(best_inner, best_angle, DEADLINE - safety - 5, beta=60.0)
        if new_side < best_side and _is_valid(best_inner, new_side, best_angle, min_sep=1e-9):
            best_side = new_side
    
    # Phase 4: Final angle polish
    if time.time() < DEADLINE - safety:
        best_side, best_angle = _polish_angle(best_inner, best_angle, DEADLINE - safety)
    
    # Phase 5: Final Nelder-Mead with high beta
    if time.time() < DEADLINE - safety - 8:
        try:
            init = best_inner.flatten().copy()
            res = minimize(
                lambda p: _objective(p, best_angle, n, beta=150.0),
                init,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if time.time() < DEADLINE - safety:
                cand = res.x.reshape(n, 3)
                cand_side = _compute_side(cand, best_angle, 0.0)
                if _is_valid(cand, cand_side, best_angle, min_sep=1e-9):
                    if cand_side < best_side:
                        best_inner = cand
                        best_side = cand_side
        except Exception:
            pass
    
    # Phase 6: Final angle polish
    if time.time() < DEADLINE - safety:
        best_side, best_angle = _polish_angle(best_inner, best_angle, DEADLINE - safety)
    
    # Final validation
    final_side = _compute_side(best_inner, best_angle, 0.0)
    if not _is_valid(best_inner, final_side, best_angle, min_sep=1e-9):
        # Fallback
        best_inner = np.array(best_seed[0], dtype=float)
        best_side = best_seed[1]
        best_angle = best_seed[2]
        final_side = best_side
    else:
        best_side = final_side
    
    inner_rows = [[float(best_inner[i, 0]), float(best_inner[i, 1]), float(best_inner[i, 2])] for i in range(n)]
    center = (0.0, 0.0)
    side = float(best_side)
    angle_deg = float(best_angle)
    
    return inner_rows, center, side, angle_deg
