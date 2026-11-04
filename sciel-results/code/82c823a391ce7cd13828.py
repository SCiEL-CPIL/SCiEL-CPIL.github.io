import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

_BASE_ANGLES = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = _BASE_ANGLES + np.deg2rad(angle_deg)
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
        t = np.clip(f / e, 0.0, 1.0)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0.0, 1.0)
        t = (b * s + f) / e
        if t < 0:
            t = 0.0
            s = np.clip(c / a, 0.0, 1.0)
        elif t > 1:
            t = 1.0
            s = np.clip((b + f) / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    t = np.clip(f / e, 0.0, 1.0)
    return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1, n2 = len(p1), len(p2)
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
    angles = _BASE_ANGLES + np.deg2rad(angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _fast_side(inner_params, outer_angle_deg):
    n = inner_params.shape[0]
    xs = inner_params[:, 0]
    ys = inner_params[:, 1]
    angles_rad = np.deg2rad(inner_params[:, 2])
    cos_a = np.cos(angles_rad)[:, np.newaxis]
    sin_a = np.sin(angles_rad)[:, np.newaxis]
    vx = _BASE_COS * cos_a - _BASE_SIN * sin_a + xs[:, np.newaxis]
    vy = _BASE_SIN * cos_a + _BASE_COS * sin_a + ys[:, np.newaxis]
    normal_angles = np.deg2rad(30.0 + outer_angle_deg + np.arange(6) * 60.0)
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = vx[:, :, np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           vy[:, :, np.newaxis] * ny[np.newaxis, np.newaxis, :]
    max_per_hex = proj.max(axis=1)
    max_over_all = max_per_hex.max(axis=0)
    return 2.0 * float(np.max(max_over_all)) / math.sqrt(3.0)

def _overlap_penalty(inner_params, penalty_weight=50.0, threshold=0.005):
    n = inner_params.shape[0]
    centers = inner_params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < 2.5:
                v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < threshold:
                    penalty += penalty_weight * (threshold - sep) ** 2
    return penalty

def _objective(params, outer_angle_deg, n):
    inner_params = params.reshape(n, 3)
    side = _fast_side(inner_params, outer_angle_deg)
    return side + _overlap_penalty(inner_params)

def _is_valid(inner_params, side, outer_angle_deg, min_sep=1e-8):
    n = inner_params.shape[0]
    side_check = _fast_side(inner_params, outer_angle_deg)
    if side_check > side + 1e-9:
        return False
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    for i in range(n):
        hex_v = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    centers = inner_params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < 2.5:
                v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
                v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True

def _hex_lattice_cluster():
    """Hexagonal lattice: center + 6 neighbors + 4 of 12 second-ring."""
    v1 = np.array([D, 0.0])
    v2 = np.array([D / 2, 1.5])
    # First ring (6 hexagons)
    first_ring = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    # Second ring: pick 4 that minimize bounding box
    # Try different selections
    candidates = [
        [(2,0), (0,2), (-2,0), (0,-2)],
        [(2,0), (1,1), (-1,1), (-2,0)],
        [(2,0), (1,1), (-1,1), (0,2)],
        [(1,1), (1,-1), (-1,1), (-1,-1)],
    ]
    best_side = float('inf')
    best_inner = None
    for second in candidates:
        selected = [(0,0)] + first_ring + second
        inner = []
        for (i, j) in selected:
            pos = i * v1 + j * v2
            inner.append([float(pos[0]), float(pos[1]), 0.0])
        arr = np.array(inner)
        # Find best outer angle
        best_s = float('inf')
        for ang in np.arange(0, 60, 0.5):
            s = _fast_side(arr, ang)
            if s < best_s:
                best_s = s
        if best_s < best_side:
            best_side = best_s
            best_inner = arr
    return best_inner, best_side

def _random_cluster():
    """Random initial cluster with hexagonal packing bias."""
    v1 = np.array([D, 0.0])
    v2 = np.array([D / 2, 1.5])
    # Start with a random connected cluster
    selected = [(0,0)]
    available = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                 (2,0), (-2,0), (0,2), (0,-2), (2,-1), (2,1), (-2,1), (-2,-1),
                 (1,1), (-1,-1), (1,-2), (-1,2)]
    np.random.shuffle(available)
    for cell in available:
        if len(selected) >= N:
            break
        if cell not in selected:
            # Check adjacency to existing
            for (ci, cj) in selected:
                di, dj = ci - cell[0], cj - cell[1]
                if abs(di) + abs(dj) <= 2:
                    selected.append(cell)
                    break
    while len(selected) < N:
        selected.append(available[len(selected) % len(available)])
    
    inner = []
    for (i, j) in selected:
        pos = i * v1 + j * v2
        angle = np.random.uniform(0, 60)
        inner.append([float(pos[0]), float(pos[1]), float(angle)])
    return np.array(inner)

def _joint_optimize(inner_arr, outer_angle, time_limit):
    """Joint optimization of inner positions/angles and outer angle."""
    n = N
    
    def full_obj(p):
        if time.time() > time_limit:
            return 1e9
        inner = p[:n*3].reshape(n, 3)
        oang = p[n*3]
        return _objective(p[:n*3], oang, n)
    
    init = np.concatenate([inner_arr.flatten(), [outer_angle]])
    best_val = full_obj(init)
    best_p = init.copy()
    
    # Multiple Nelder-Mead restarts
    for restart in range(3):
        if time.time() > time_limit - 5:
            break
        try:
            res = minimize(full_obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p[:n*3].reshape(n,3), best_p[n*3], best_val

def _polish_angle(inner_arr, initial_angle, time_limit):
    best_side = _fast_side(inner_arr, initial_angle)
    best_angle = initial_angle
    for step, span in [(0.5, 5.0), (0.05, 0.5), (0.005, 0.05)]:
        for angle_deg in np.arange(best_angle - span, best_angle + span, step):
            if time.time() > time_limit:
                break
            side = _fast_side(inner_arr, angle_deg)
            if side < best_side:
                best_side, best_angle = side, angle_deg
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _fast_side(inner_arr, a),
                bounds=(best_angle - 0.01, best_angle + 0.01),
                method='bounded', options={'xatol': 1e-12}
            )
            if res.success and res.fun < best_side:
                best_side, best_angle = res.fun, res.x
        except Exception:
            pass
    return best_side, best_angle

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    n = N
    
    # Generate seed candidates
    lattice_inner, lattice_side = _hex_lattice_cluster()
    
    # Find best angle for lattice
    best_angle = 0.0
    for ang in np.arange(0, 60, 0.1):
        s = _fast_side(lattice_inner, ang)
        if s < lattice_side:
            lattice_side = s
            best_angle = ang
    
    best_inner = lattice_inner.copy()
    best_side = lattice_side
    best_ang = best_angle
    
    # Phase 1: Joint optimization (inner positions + outer angle)
    if time.time() < DEADLINE - safety_margin - 30:
        try:
            cand, ang, val = _joint_optimize(best_inner, best_ang, DEADLINE - safety_margin - 10)
            cand_side = _fast_side(cand, ang)
            if _is_valid(cand, cand_side, ang, min_sep=1e-8) and cand_side < best_side:
                best_inner, best_side, best_ang = cand, cand_side, ang
        except Exception:
            pass
    
    # Phase 2: Angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_ang = _polish_angle(best_inner, best_ang, DEADLINE - safety_margin)
    
    # Phase 3: Another joint optimization round
    if time.time() < DEADLINE - safety_margin - 20:
        try:
            cand, ang, val = _joint_optimize(best_inner, best_ang, DEADLINE - safety_margin - 5)
            cand_side = _fast_side(cand, ang)
            if _is_valid(cand, cand_side, ang, min_sep=1e-8) and cand_side < best_side:
                best_inner, best_side, best_ang = cand, cand_side, ang
        except Exception:
            pass
    
    # Phase 4: Final angle polish
    if time.time() < DEADLINE - safety_margin:
        best_side, best_ang = _polish_angle(best_inner, best_ang, DEADLINE - safety_margin)
    
    # Phase 5: Random restarts if time permits
    if time.time() < DEADLINE - safety_margin - 15:
        for _ in range(2):
            if time.time() > DEADLINE - safety_margin - 5:
                break
            try:
                rand_inner = _random_cluster()
                # Quick angle find
                ra = 0.0
                rs = float('inf')
                for ang in np.arange(0, 60, 1.0):
                    s = _fast_side(rand_inner, ang)
                    if s < rs:
                        rs = s
                        ra = ang
                if rs < best_side + 0.5:  # Only optimize if promising
                    cand, ang, val = _joint_optimize(rand_inner, ra, DEADLINE - safety_margin - 3)
                    cand_side = _fast_side(cand, ang)
                    if _is_valid(cand, cand_side, ang, min_sep=1e-8) and cand_side < best_side:
                        best_inner, best_side, best_ang = cand, cand_side, ang
            except Exception:
                pass
    
    # Final validation
    final_side = _fast_side(best_inner, best_ang)
    if not _is_valid(best_inner, final_side, best_ang, min_sep=1e-8):
        # Fallback to lattice
        best_inner = lattice_inner.copy()
        best_side = lattice_side
        best_ang = 0.0
        for ang in np.arange(0, 60, 0.1):
            s = _fast_side(best_inner, ang)
            if s < best_side:
                best_side = s
                best_ang = ang
    else:
        best_side = final_side
    
    inner_rows = [[float(best_inner[i, 0]), float(best_inner[i, 1]), float(best_inner[i, 2])]
                  for i in range(n)]
    return inner_rows, (0.0, 0.0), float(best_side), float(best_ang)
