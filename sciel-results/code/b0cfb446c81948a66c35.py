import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
    """Returns minimum overlap (negative if overlapping) using SAT."""
    min_overlap = float('inf')
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return -min_overlap

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, eps=1e-9):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_fast(verts[i], verts[j], norms[i], norms[j])
            if d < -eps:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > 12:
        pts = pts[:12]
    elif len(pts) < 12:
        while len(pts) < 12:
            pts.append([0, 0, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_packed_cluster(gap=0.0):
    """Create a tightly packed cluster of 12 hexagons."""
    pts = []
    sqrt3 = np.sqrt(3)
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_strip(gap=0.0):
    """Create a strip configuration."""
    pts = []
    sqrt3 = np.sqrt(3)
    for i in range(12):
        x = i * (sqrt3 + gap)
        y = 0 if i % 2 == 0 else 0.75
        pts.append([x, y, 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _evaluate_config(inner, outer_rot_deg):
    """Evaluate a configuration, returning (L, is_valid)."""
    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
    ox = np.mean(all_pts[:, 0])
    oy = np.mean(all_pts[:, 1])
    
    if not _check_disjoint(inner):
        return float('inf'), False
    
    all_pts_centered = all_pts.copy()
    all_pts_centered[:, 0] -= ox
    all_pts_centered[:, 1] -= oy
    L = _outer_hex_side_fast(all_pts_centered, outer_rot_deg)
    return L, True

def _physics_relax(pts, max_iter=500, dt=0.01, cooling=0.995):
    """Physics-based relaxation to resolve collisions."""
    n = len(pts)
    verts_list = []
    norms_list = []
    for i in range(n):
        x, y, rot = pts[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    forces = np.zeros((n, 3))
    temp = 1.0
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        forces = np.zeros((n, 3))
        for i in range(n):
            for j in range(i + 1, n):
                d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
                if d < 0:
                    overlap = -d
                    dx = pts[j, 0] - pts[i, 0]
                    dy = pts[j, 1] - pts[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-9:
                        fx = (dx / dist) * overlap * 10.0
                        fy = (dy / dist) * overlap * 10.0
                    else:
                        fx = overlap * 10.0
                        fy = 0.0
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        for i in range(n):
            forces[i, 0] += np.random.randn() * temp * 0.1
            forces[i, 1] += np.random.randn() * temp * 0.1
            forces[i, 2] += np.random.randn() * temp * 0.05
        
        for i in range(n):
            pts[i, 0] += forces[i, 0] * dt
            pts[i, 1] += forces[i, 1] * dt
            pts[i, 2] += forces[i, 2] * dt
            verts_list[i] = _hex_vertices(pts[i, 0], pts[i, 1], 1.0, pts[i, 2])
            norms_list[i] = _hex_edge_normals(pts[i, 2])
        
        temp *= cooling
    
    return pts

def _objective(p, t0, deadline, beta=100.0):
    """
    Objective function: Minimize outer side length + penalty for overlaps.
    p: [x1, y1, r1, ..., x12, y12, r12, ox, oy, oang]
    """
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    # Calculate overlaps
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlaps.append(-d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        scaled = 2.0 * p_arr / beta
        max_s = np.max(scaled)
        if max_s > 500:
            penalty = np.sum(p_arr**2)
        else:
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            penalty = beta * lse
    else:
        penalty = 0.0

    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + 100.0 * penalty

def _objective_nm(p, t0, deadline, beta=100.0):
    """
    Objective for Nelder-Mead with tighter penalty scaling.
    """
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlaps.append(-d)
                
    if overlaps:
        p_arr = np.array(overlaps)
        scaled = 3.0 * p_arr / beta
        max_s = np.max(scaled)
        if max_s > 500:
            penalty = np.sum(p_arr**2)
        else:
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            penalty = beta * lse
    else:
        penalty = 0.0

    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + 100.0 * penalty

def _polish(params, t0, deadline):
    """
    Multi-stage polishing to ensure high precision.
    """
    if time.time() - t0 > deadline - 10:
        return params
    
    # Stage 1: L-BFGS-B with moderate penalty
    try:
        res = minimize(
            _objective,
            params,
            args=(t0, deadline - 10, 50.0),
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 5000
            }
        )
        if res.fun < _objective(params, t0, deadline - 10, 50.0):
            params = res.x
    except Exception:
        pass
    
    # Stage 2: L-BFGS-B with higher penalty to enforce strict disjointness
    if time.time() - t0 < deadline - 5:
        try:
            res = minimize(
                _objective,
                params,
                args=(t0, deadline - 5, 200.0),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 5000
                }
            )
            if res.fun < _objective(params, t0, deadline - 5, 200.0):
                params = res.x
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final local refinement if time permits
    if time.time() - t0 < deadline - 2:
        try:
            res = minimize(
                _objective_nm,
                params,
                args=(t0, deadline - 2, 500.0),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-14,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res.fun < _objective_nm(params, t0, deadline - 2, 500.0):
                params = res.x
        except Exception:
            pass
            
    return params

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Multiple initial configurations with different rotation patterns
    configs = [
        _make_hex_grid(4, 3, gap=0.1),
        _make_hex_grid(3, 4, gap=0.1),
        _make_hex_grid(2, 6, gap=0.1),
        _make_hex_grid(6, 2, gap=0.1),
        _make_ring(gap=0.1),
        _make_packed_cluster(gap=0.1),
        _make_strip(gap=0.1),
        _make_hex_grid(4, 3, rot_offset=30, gap=0.1),
        _make_hex_grid(3, 4, rot_offset=30, gap=0.1),
        _make_hex_grid(2, 6, rot_offset=30, gap=0.1),
    ]
    
    # Add random configurations with varied rotations
    np.random.seed(42)
    for _ in range(15):
        pts = np.random.randn(12, 3) * 3.0
        rot_pattern = np.random.choice(['uniform', 'alternating', 'radial'])
        if rot_pattern == 'uniform':
            pts[:, 2] = np.random.uniform(0, 60, 12)
        elif rot_pattern == 'alternating':
            pts[:, 2] = np.where(np.arange(12) % 2 == 0, 0.0, 30.0) + np.random.uniform(-5, 5, 12)
        else:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            pts[:, 2] = np.degrees(angles) % 60
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    best_L = float('inf')
    best_inner = None
    best_center = None
    best_angle = None
    best_params = None
    
    # Phase 1: Physics relaxation on each config, then evaluate
    for idx, cfg in enumerate(configs):
        if time_left() < 20:
            break
        
        relaxed = _physics_relax(cfg.copy(), max_iter=300, dt=0.01, cooling=0.995)
        
        for oang in [0.0, 30.0]:
            if time_left() < 5:
                break
            
            L, valid = _evaluate_config(relaxed, oang)
            if valid and L < best_L:
                best_L = L
                best_inner = relaxed.copy()
                all_pts = np.vstack([_hex_vertices(relaxed[i,0], relaxed[i,1], 1.0, relaxed[i,2]) for i in range(12)])
                best_center = (np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1]))
                best_angle = oang
                best_params = np.concatenate((relaxed.flatten(), [best_center[0], best_center[1], oang]))
    
    # Phase 2: Local optimization from best config using multi-stage polish
    if best_params is not None and time_left() > 40:
        polished = _polish(best_params.copy(), t0, time_left() - 5)
        
        inner_opt = polished[:36].reshape(12, 3)
        if _check_disjoint(inner_opt):
            ox_opt = polished[36]
            oy_opt = polished[37]
            oang_opt = polished[38]
            all_pts_opt = np.vstack([_hex_vertices(inner_opt[i,0], inner_opt[i,1], 1.0, inner_opt[i,2]) for i in range(12)])
            all_pts_opt[:, 0] -= ox_opt
            all_pts_opt[:, 1] -= oy_opt
            L_opt = _outer_hex_side_fast(all_pts_opt, oang_opt)
            if L_opt < best_L:
                best_L = L_opt
                best_inner = inner_opt
                best_center = (ox_opt, oy_opt)
                best_angle = oang_opt
                best_params = polished.copy()
    
    # Phase 3: Try more random starts with physics + optimization
    if time_left() > 25:
        for _ in range(8):
            if time_left() < 10:
                break
            
            pts = np.random.randn(12, 3) * 2.5
            rot_pattern = np.random.choice(['uniform', 'alternating', 'radial'])
            if rot_pattern == 'uniform':
                pts[:, 2] = np.random.uniform(0, 60, 12)
            elif rot_pattern == 'alternating':
                pts[:, 2] = np.where(np.arange(12) % 2 == 0, 0.0, 30.0) + np.random.uniform(-5, 5, 12)
            else:
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
                pts[:, 2] = np.degrees(angles) % 60
            pts[:, 0] -= np.mean(pts[:, 0])
            pts[:, 1] -= np.mean(pts[:, 1])
            
            relaxed = _physics_relax(pts.copy(), max_iter=200, dt=0.01, cooling=0.995)
            
            for oang in [0.0, 30.0]:
                if time_left() < 5:
                    break
                
                L, valid = _evaluate_config(relaxed, oang)
                if valid and L < best_L:
                    best_L = L
                    best_inner = relaxed.copy()
                    all_pts = np.vstack([_hex_vertices(relaxed[i,0], relaxed[i,1], 1.0, relaxed[i,2]) for i in range(12)])
                    best_center = (np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1]))
                    best_angle = oang
                    best_params = np.concatenate((relaxed.flatten(), [best_center[0], best_center[1], oang]))
    
    # Final polish if time allows
    if best_params is not None and time_left() > 15:
        polished = _polish(best_params.copy(), t0, time_left() - 3)
        inner_opt = polished[:36].reshape(12, 3)
        if _check_disjoint(inner_opt):
            ox_opt = polished[36]
            oy_opt = polished[37]
            oang_opt = polished[38]
            all_pts_opt = np.vstack([_hex_vertices(inner_opt[i,0], inner_opt[i,1], 1.0, inner_opt[i,2]) for i in range(12)])
            all_pts_opt[:, 0] -= ox_opt
            all_pts_opt[:, 1] -= oy_opt
            L_opt = _outer_hex_side_fast(all_pts_opt, oang_opt)
            if L_opt < best_L:
                best_L = L_opt
                best_inner = inner_opt
                best_center = (ox_opt, oy_opt)
                best_angle = oang_opt
    
    if best_inner is None:
        # Fallback: use a known valid configuration
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    inner = []
    for i in range(12):
        x = best_inner[i, 0]
        y = best_inner[i, 1]
        r = best_inner[i, 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_angle = float(best_angle)
    
    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
